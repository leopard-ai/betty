# Copyright Sang Keun Choe
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
import torch.distributed as dist

from betty.configs import EngineConfig
from betty.logging import logger
from betty.misc.early_stopping import EarlyStopping
from betty.utils import log_from_loss_dict


class Engine:
    """
    ``Engine`` handles a dataflow graph based on the user-provided hierarchical problem
    dependencies. It also provides a primitive for executing multilevel optimization.
    """

    def __init__(self, problems, config=None, dependencies=None, env=None):
        # config
        self.config = config if config is not None else EngineConfig()

        # step counters
        self.train_iters = 0
        self.valid_step = 0
        self.global_step = 0

        # logger
        self.logger_type = None
        self.logger = None

        # problem
        self.problems = problems
        self.leaves = []

        # dependencies
        self.dependencies = dependencies

        # env
        self.env = env

        # distributed
        self._strategy = None
        self._backend = None
        self._world_size = 0
        self._rank = 0
        self._local_rank = 0

        # early stopping
        self.early_stopping = None

        # roll back
        self._roll_back = False

        # initialize
        self.initialize()

    def parse_config(self):
        """
        Parse EngineConfig.
        """
        self.train_iters = self.config.train_iters
        self.valid_step = self.config.valid_step

        self.logger_type = self.config.logger_type

        self._roll_back = self.config.roll_back

        self._strategy = self.config.strategy
        self._backend = self.config.backend

        if self.config.early_stopping:
            self.early_stopping = EarlyStopping(
                metric=self.config.early_stopping_metric,
                mode=self.config.early_stopping_mode,
                tolerance=self.config.early_stopping_tolerance,
            )

    def train_step(self):
        """
        Running one-step gradient descent for all leaf problems.
        """
        for leaf in self.leaves:
            leaf.step(global_step=self.global_step)

    def run(self):
        """
        Execute multilevel optimization by running gradient descent for leaf problems.
        """
        self.train()
        for it in range(1, self.train_iters + 1):
            self.global_step += 1
            self.train_step()

            if it % self.valid_step == 0 and self.do_validation():
                self.eval()
                validation_stats = self.validation() or {}
                log_loss = log_from_loss_dict(validation_stats)
                self.logger.info(
                    f"[Validation] [Global Step {self.global_step}] " f"{log_loss}"
                )
                self.logger.log(
                    validation_stats, tag="validation", step=self.global_step
                )
                self.train()

                # early stopping
                if self.early_stopping is not None:
                    stop = self.early_stopping(validation_stats)
                    if stop:
                        self.logger.info("Early stopping is executed!")
                        break

    def initialize(self):
        """
        Initialize dependencies (computational graph) between problems.
        """
        # Parse config
        self.parse_config()

        # initialize distributed training
        dist_dict = self.initialize_distributed()

        # initialize logger
        self.logger = logger(logger_type=self.logger_type)
        if self.is_rank_zero():
            self.logger.info("Initializing Multilevel Optimization...\n")
        start = time.time()

        # parse dependency
        self.parse_dependency()

        # set problem attributes
        for problem in self.problems:
            self.set_problem_attr(problem)

        # problem initialization
        for problem in self.problems:
            problem.add_logger(self.logger)
            problem.configure_distributed_training(dist_dict)
            problem.configure_roll_back(self._roll_back)
            problem.initialize()
            if self.env is not None:
                problem.add_env(self.env)

        end = time.time()
        if self.is_rank_zero():
            self.logger.info(f"Time spent on initialization: {end-start:.3f} (s)\n")

    def initialize_distributed(self):
        """
        Initialize distributed training.
        """
        if self._strategy in ["distributed", "zero", "fsdp"]:
            dist.init_process_group(backend=self._backend)

            self._world_size = dist.get_world_size()
            assert self._world_size > 1
            self._rank = dist.get_rank()

            device_count = torch.cuda.device_count()
            self._local_rank = self._rank % device_count

        dist_dict = {}
        dist_dict["strategy"] = self._strategy
        dist_dict["backend"] = self._backend
        dist_dict["world_size"] = self._world_size
        dist_dict["rank"] = self._rank
        dist_dict["local_rank"] = self._local_rank

        return dist_dict

    def train(self):
        """
        Set all problems in multilevel optimization to the train mode.
        """
        for problem in self.problems:
            problem.train()

    def eval(self):
        """
        Set all problems in multilevel optimization to the eval mode.
        """
        for problem in self.problems:
            problem.eval()

    def check_leaf(self, problem):
        """
        Check whether the given ``problem`` is a leaf problem or not.

        :param problem: Problem in multilevel optimization
        :type problem: Problem
        :return: True or False
        :rtype: bool
        """
        for _, value_list in self.dependencies["l2u"].items():
            if problem in set(value_list):
                return False

        return True

    def find_paths(self, src, dst):
        """
        Find all paths from ``src`` to ``dst`` with a modified depth-first search algorithm.

        :param src: The end point of the upper-to-lower edge.
        :type src: Problem
        :param dst: The start point of the upper-to-lower edge.
        :type dst: Problem
        :return: List of all paths from ``src`` to ``dst``.
        """
        results = []
        path = [src]
        self.dfs(src, dst, path, results)
        assert len(results) > 0, f"No path from {src.name} to {dst.name}!"

        for i, _ in enumerate(results):
            results[i].reverse()
            results[i].append(dst)

        return results

    def dfs(self, src, dst, path, results):
        if src is dst:
            assert len(path) > 1
            result = [node for node in path]
            results.append(result)
        elif src not in self.dependencies["l2u"]:
            return
        else:
            for adj in self.dependencies["l2u"][src]:
                path.append(adj)
                self.dfs(adj, dst, path, results)
                path.pop()

    def parse_dependency(self):
        """
        Parse user-provided ``u2l`` and ``l2u`` dependencies to figure out 1) topological order for
        multilevel optimization execution, and 2) backpropagation path(s) for each problem. A
        modified depth-first search algorithm is used.
        """
        # Parse upper-to-lower dependency
        for key, value_list in self.dependencies["u2l"].items():
            for value in value_list:

                # find all paths from low to high for backpropagation
                paths = self.find_paths(src=value, dst=key)
                key.add_paths(paths)

        # Parse lower-to-upper dependency
        for key, value_list in self.dependencies["l2u"].items():
            for value in value_list:

                # add value problem to parents of key problem for backpropgation
                key.add_parent(value)
                value.add_child(key)

        # Parse problems
        for problem in self.problems:
            if self.check_leaf(problem):
                problem.leaf = True
                self.leaves.append(problem)

    def set_dependency(self, dependencies):
        self.dependencies = dependencies
        self.leaves = []

        # clear existing dependencies
        for problem in self.problems:
            problem.leaf = False
            problem.clear_dependencies()

        self.parse_dependency()

    def set_problem_attr(self, problem):
        """
        Set class attribute for the given ``problem`` based on their names

        :param problem: Problem in multilevel optimization
        :type problem: Problem
        :return: ``problem`` name
        :rtype: str
        """
        name = problem.name

        # set attribute for Engine
        assert not hasattr(self, name), f"Problem already has a problelm named {name}!"
        setattr(self, name, problem)

        # set attribute for Problems
        for prob in self.problems:
            if prob != problem:
                assert not hasattr(problem, name)
                setattr(prob, name, problem)

        # set attribute for Env
        if self.env is not None:
            setattr(self.env, name, problem)

        return name

    def do_validation(self):
        """
        Check whether to run validation.
        """
        if self.is_implemented("validation") and self.is_rank_zero():
            return True
        return False

    def is_rank_zero(self):
        """
        Check whether the current process is rank 0
        """
        return self._rank == 0

    def is_implemented(self, fn_name):
        """
        Check whether ``fn_name`` method is implemented in the class.

        :param fn_name: class method name
        :type fn_name: str
        :rtype: bool
        """
        return callable(getattr(self, fn_name, None))
