import copy
import typing

from betty.utils import get_multiplier
from betty.config_template import EngineConfig


class Engine:
    def __init__(self, config, problems, dependencies=None):
        # config
        self.config = config if config is not None else EngineConfig()
        self.train_iters = 0
        self.valid_step = 0

        # problem
        self.problems = problems
        self._problem_name_dict = {}
        self.leaves = []

        # dependencies
        self.dependencies = dependencies

        # initialize
        self.initialize()

    def parse_config(self):
        self.train_iters = self.config.train_iters
        self.valid_step = self.config.valid_step

    def train_step(self):
        for leaf in self.leaves:
            leaf.step()

    def run(self):
        self.train()
        for it in range(1, self.train_iters + 1):
            self.train_step()

            if it % self.valid_step == 0:
                if self.is_implemented('validation'):
                    self.eval()
                    self.validation()
                    self.train()

    def initialize(self):
        """[summary]
        Initialize dependencies (computational graph) between problems.
        """
        # Parse config
        self.parse_config()

        # Parse dependency
        self.parse_dependency()

        # check & set multiplier for each problem
        for problem in self.problems:
            #multiplier = get_multiplier(problem)
            #problem.multiplier = multiplier
            problem.initialize()

    def train(self):
        for problem in self.problems:
            problem.train()

    def eval(self):
        for problem in self.problems:
            problem.eval()

    def check_leaf(self, problem):
        for _, value_list in self.dependencies['l2h'].items():
            if problem in set(value_list):
                return False

        return True

    def find_paths(self, src, dst):
        results = []
        path = [src]
        self.dfs(src, dst, path, results)
        assert len(results) > 0, f'No path from {src.name} to {dst.name}!'

        for i in range(len(results)):
            results[i].reverse()
            results[i].append(dst)

        return results

    def dfs(self, src, dst, path, results):
        if src is dst:
            assert len(path) > 1
            result = [node for node in path]
            results.append(result)
        else:
            for adj in self.dependencies['l2h'][src]:
                path.append(adj)
                self.dfs(adj, dst, path, results)
                path.pop()

    def parse_dependency(self, set_attr=True):
        # Set dependencies for high-to-low dependencies
        for key, value_list in self.dependencies['h2l'].items():
            for value in value_list:
                # set the problelm attribute for key problem in value problem
                if set_attr:
                    value.set_problem_attr(key)

                # find all paths from low to high for backpropagation
                paths = self.find_paths(src=value, dst=key)
                key.add_paths(paths)

        # Set dependencies for low-to-high dependencies
        for key, value_list in self.dependencies['l2h'].items():
            for value in value_list:
                # set the problelm attribute for key problem in value problem
                if set_attr:
                    value.set_problem_attr(key)

                # add value problem to parents of key problem for backpropgation
                key.add_parent(value)
                value.add_child(key)

        # Parse problems
        for problem in self.problems:
            if set_attr:
                self.set_problem_attr(problem)
            if self.check_leaf(problem):
                problem.leaf = True
                self.leaves.append(problem)

    def set_dependency(self, dependencies, set_attr=False):
        self.dependencies = dependencies
        self.leaves = []

        # clear existing dependencies
        for problem in self.problems:
            problem.leaf = False
            problem.clear_dependencies()

        self.parse_dependency(set_attr=set_attr)

    def set_problem_attr(self, problem):
        """[summary]
        Set class attributed for parent/children problems based on their names
        """
        name = problem.name
        if name not in self._problem_name_dict:
            assert not hasattr(self, name), f'Problem already has an attribute named {name}!'
            self._problem_name_dict[name] = 0
            setattr(self, name, problem)
        elif self._problem_name_dict[name] == 0:
            # rename first problem
            first_problem = getattr(self, name)
            delattr(self, name)
            setattr(self, name + '_0', first_problem)

            self._problem_name_dict[name] += 1
            name = name + '_' + str(self._problem_name_dict[name])
            setattr(self, name, problem)
        else:
            self._problem_name_dict[name] += 1
            name = name + '_' + str(self._problem_name_dict[name])
            setattr(self, name, problem)

        return name

    def is_implemented(self, fn_name):
        return callable(getattr(self, fn_name, None))
