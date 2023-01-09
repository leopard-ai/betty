# Copyright Sang Keun Choe
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import abc

import torch
import torch.distributed as dist

from betty.patch.data_loader import get_distributed_data_loader
from betty.patch.optimizer import patch_optimizer
from betty.patch.scheduler import patch_scheduler
from betty.configs import Config
from betty.hypergradient import get_grads
from betty.utils import convert_tensor, log_from_loss_dict


class Problem:
    """
    This is the base class for an optimization problem in multilevel optimization.
    Specifically, each problem is defined by the parameter (or module), the sets of the upper
    and lower constraining problems, the dataset, the loss function, the optimizer, and other
    optimization configurations (e.g. best-response Jacobian calculation algorithm, number of
    unrolling steps, etc.).
    """

    def __init__(
        self,
        name,
        config=None,
        module=None,
        optimizer=None,
        scheduler=None,
        train_data_loader=None,
        device=None,
    ):
        # basic configurations
        self._name = name
        self._config = config if config is not None else Config()

        # device
        self.device = device

        # distributed
        self._strategy = None
        self.accelerator = None
        self._distributed = False
        self._backend = None
        self._world_size = None
        self._rank = None
        self._local_rank = None

        # computation graph depedency
        self._parents = []
        self._children = []
        self._paths = []

        # data loader
        self.train_data_loader = train_data_loader
        self.train_data_iterator = None
        self.cur_batch = None
        self.epoch_counter = None

        # module
        self.module = module

        # optimizer & lr scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler

        # environment
        self.env = None

        # fp16 scaler
        self._fp16 = config.fp16
        self.scaler = None
        if self._fp16:
            self.initial_dynamic_scale = config.initial_dynamic_scale
            self.scale_factor = config.scale_factor

        # gradient accumulation
        self.gas = config.gradient_accumulation

        # gradient clipping
        self.gradient_clipping = config.gradient_clipping

        # warmup
        self.warmup_steps = config.warmup_steps

        # logger
        self.logger = None
        self.log_step = config.log_step
        self.log_local_step = config.log_local_step

        # step counter
        self._count = 0
        self._global_step = 0

        # misc
        self._leaf = False
        self._first_order = False
        self._retain_graph = config.retain_graph
        self._allow_unused = config.allow_unused
        self._unroll_steps = config.unroll_steps
        self._roll_back = False
        self._inner_loop_start = True
        self._training = True
        self.ready = None

    def initialize(self):
        """
        ``initialize`` patches/sets up module, optimizer, data loader, etc. after compiling a
        user-provided configuration (e.g., fp16 training, iterative differentiation)
        """
        # configure device
        self.configure_device()

        # initialize update ready to False
        self.ready = [False for _ in range(len(self._children))]

        # compile parents configurations
        first_order = []
        for problem in self._parents:
            parent_config = problem.config
            first_order.append(parent_config.first_order)
        self._first_order = all(first_order)

        # set inner_loop_start to True
        self._inner_loop_start = True

        # set up data loader
        if self.is_implemented("configure_train_data_loader"):
            if self.train_data_loader is None:
                self.train_data_loader = self.configure_train_data_loader()
        if self.train_data_loader is not None:
            if not isinstance(self.train_data_loader, tuple):
                self.train_data_loader = (self.train_data_loader,)
        else:
            assert self.is_implemented("get_batch")

        # set up module
        if self.is_implemented("configure_module"):
            if self.module is None:
                self.module = self.configure_module()
        assert self.module is not None, "Module must be specified!"

        # set up optimizer
        if self.is_implemented("configure_optimizer"):
            if self.optimizer is None:
                self.optimizer = self.configure_optimizer()

        # set up lr scheduler
        if self.is_implemented("configure_scheduler"):
            if self.scheduler is None:
                self.scheduler = self.configure_scheduler()

        # set up fp16 training
        if self._is_default_fp16():
            assert torch.cuda.is_available()
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=self.initial_dynamic_scale, growth_factor=self.scale_factor
            )

        # patch module, optimizer, data loader, and scheduler
        self.patch_module_optimizer_loader()

        # make train_data_loader as iterator
        if self.train_data_loader is not None:
            self.train_data_iterator = []
            self.epoch_counter = []
            for train_data_loader in self.train_data_loader:
                self.train_data_iterator.append(iter(train_data_loader))
                self.epoch_counter.append(0)

        # Logging INFO
        path_str = [[node.name for node in path] for path in self._paths]
        children_str = [node.name for node in self._children]
        parents_str = [node.name for node in self._parents]
        if self.is_rank_zero():
            self.logger.info("*** Problem Information ***")
            self.logger.info(f"Name: {self._name}")
            self.logger.info(f"Uppers: {parents_str}")
            self.logger.info(f"Lowers: {children_str}")
            self.logger.info(f"Paths: {path_str}\n")

    def patch_module_optimizer_loader(self):
        """
        We patch module, optimizer, data loader, and lr scheduler for device placement,
        distributed training, zero optimizer, fsdp, etc.
        """
        # patch module
        self.module.to(self.device)
        if self._strategy in ["distributed", "zero"]:
            self.synchronize_params(self.parameters())
            self.module = torch.nn.parallel.DistributedDataParallel(
                module=self.module,
                gradient_as_bucket_view=True,
            )
        elif self._strategy == "fsdp":
            if self.is_rank_zero():
                self.logger.warning("FSDP requires PyTorch version >= 1.12")
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            self.synchronize_params(self.parameters())
            self.module = FSDP(self.module, device_id=self.device)
        elif self._strategy == "accelerate":
            self.module = self.accelerator.prepare(self.module)

        # patch optimizer
        params = self.trainable_parameters()
        if self.is_implemented("param_groups") and self._strategy != "fsdp":
            params = self.param_groups()
        is_zero = True if self._strategy == "zero" else False
        self.optimizer = patch_optimizer(self.optimizer, params, is_zero)

        # patch scheduler
        if self.scheduler is not None:
            self.scheduler = patch_scheduler(self.scheduler, self.optimizer)
            if self._strategy == "accelerate":
                self.scheduler = self.accelerator.prepare(self.scheduler)

        if self._is_default_fp16() and self._strategy == "fsdp":
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(
                init_scale=self.initial_dynamic_scale, growth_factor=self.scale_factor
            )

        # accelerate optimizer & scheduler patch
        if self._strategy == "accelerate":
            if self.scheduler is None:
                self.optimizer = self.accelerator.prepare(self.optimizer)
            else:
                self.optimizer, self.scheduler = self.accelerator.prepare(
                    self.optimizer, self.scheduler
                )

        # patch data loader
        if self.train_data_loader is not None:
            if self._strategy in ["distributed", "zero", "fsdp"]:
                self.train_data_loader = [
                    get_distributed_data_loader(
                        loader, world_size=self._world_size, rank=self._rank
                    )
                    for loader in self.train_data_loader
                ]
            elif self._strategy == "accelerate":
                self.train_data_loader = [
                    self.accelerator.prepare(data_loader)
                    for data_loader in self.train_data_loader
                ]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Users define how forward (or call) function is defined for the problem here.
        """
        return self.module(*args, **kwargs)

    @abc.abstractmethod
    def training_step(self, batch):
        """
        Users define the loss function of the problem here.
        """
        raise NotImplementedError

    def training_step_exec(self, batch):
        if self._is_default_fp16():
            with torch.cuda.amp.autocast():
                return self.training_step(batch)
        else:
            return self.training_step(batch)

    def one_step_descent(self, batch=None):
        # load data
        if batch is None:
            self.cur_batch = self.get_batch()
            batch = self.cur_batch

        # calculate loss
        loss, loss_dict = self.get_loss(batch)

        # calculate gradient (a.k.a backward)
        self.backward(
            loss=loss,
            params=self.trainable_parameters(),
            paths=self._paths,
            create_graph=not self._first_order,
            retain_graph=self._retain_graph,
            allow_unused=self._allow_unused,
        )

        # calculate parameter update
        if self._count % self.gas == 0:
            self.optimizer_step()

            # param callback (e.g., parameter clipping)
            if self.is_implemented("param_callback"):
                self.param_callback(self.trainable_parameters())

            if self._strategy != "default" and self._count % (self.gas * 20) == 0:
                self.synchronize_params(self.trainable_parameters())

            # zero-out grad
            self.zero_grad()

        return loss_dict

    def step_normal(self, global_step=None):
        if self.check_ready():
            # loop start
            if self._inner_loop_start:
                if self.is_implemented("on_inner_loop_start"):
                    self.on_inner_loop_start()
                self._inner_loop_start = False

                # copy current parameters, buffers, optimizer states
                if self._roll_back:
                    self.cache_states()

            # increase count (local step)
            if self._training:
                self._count += 1

            # one step grdient descent
            loss_dict = self.one_step_descent()

            # lr scheduler step
            if self.scheduler is not None and not self._roll_back:
                self.scheduler.step()

            # logging
            if (
                self.log_step > 0
                and self._count % self.log_step == 0
                and self.is_rank_zero()
            ):
                self.log(loss_dict, global_step)

            # call parent step_normal after unrolling
            if (
                self._training
                and self._count % (self._unroll_steps * self.gas) == 0
                and self._count > self.warmup_steps
            ):
                for problem in self._parents:
                    idx = problem.children.index(self)
                    problem.ready[idx] = True
                    problem.step_normal(global_step=global_step)

                self._inner_loop_start = True

            self.ready = [False for _ in range(len(self._children))]

    def step_after_roll_back(self):
        if self.check_ready() and self._training:
            if self._roll_back:
                # recover from cached states
                self.recover_states()

                # one step gradient step
                _ = self.one_step_descent(batch=self.cur_batch)

                # lr scheduler
                if self.scheduler is not None:
                    self.scheduler.step()

                # call parent step_after_roll_back
                for problem in self._parents:
                    idx = problem.children.index(self)
                    problem.ready[idx] = True
                    problem.step_after_roll_back()

            self.ready = [False for _ in range(len(self._children))]

    def step(self, global_step=None):
        """
        ``step`` method abstracts a one-step gradient descent update with four sub-steps:
        1) data loading, 2) cost calculation, 3) gradient calculation, and 4) parameter update.
        It also calls upper-level problems' step methods after unrolling gradient steps based on
        the hierarchical dependency graph.

        :param global_step: global step of the whole multilevel optimization. Defaults to None.
        :type global_step: int, optional
        """
        self._global_step = global_step
        self.step_normal(global_step=global_step)
        if (
            self._count % (self._unroll_steps * self.gas) == 0
            and self._count > self.warmup_steps
        ):
            self.step_after_roll_back()

    def get_batch(self):
        """
        Load training batch from the user-provided data loader

        :return: New training batch
        :rtype: Any
        """
        batch = tuple(
            self.get_batch_single_loader(i) for i in range(len(self.train_data_loader))
        )

        return batch[0] if len(batch) == 1 else batch

    def get_batch_single_loader(self, idx):
        """
        Load training batch from one of the user-provided data loader(s)

        :return: New training batch
        :rtype: Any
        """
        data_iterator = self.train_data_iterator[idx]
        try:
            batch = next(data_iterator)
        except StopIteration:
            if idx == 0:
                self.on_epoch_end_exec()
            self.epoch_counter[idx] += 1
            train_data_loader = self.train_data_loader[idx]
            if self._strategy in ["distributed", "zero", "fsdp"]:
                train_data_loader.set_epoch(self.epoch_counter[idx])
            self.train_data_iterator[idx] = iter(train_data_loader)
            batch = next(self.train_data_iterator[idx])
        if not isinstance(batch, dict):
            batch = tuple(
                convert_tensor(value, self.device, self._is_default_fp16())
                for value in batch
            )
        else:
            for key, value in batch.items():
                batch[key] = convert_tensor(value, self.device, self._is_default_fp16())

        return batch

    def get_loss(self, batch):
        """
        Calculate loss and log metrics for the current batch based on the user-defined loss
        function.

        :return: loss and log metrics (e.g. classification accuracy)
        :rtype: dict
        """
        maybe_loss_dict = self.training_step_exec(batch)
        is_dict = isinstance(maybe_loss_dict, dict)
        loss = maybe_loss_dict["loss"] if is_dict else maybe_loss_dict
        loss_no_scale = loss.item()
        if self._is_default_fp16():
            loss = self.scaler.scale(loss)
        loss = loss / self.gas

        # construct loss dict
        loss_dict = {"loss": loss_no_scale}
        if is_dict:
            for key, value in maybe_loss_dict.items():
                if key != "loss":
                    loss_dict[key] = value

        return loss, loss_dict

    def backward(
        self,
        loss,
        params,
        paths,
        create_graph=False,
        retain_graph=True,
        allow_unused=True,
    ):
        """
        Calculate the gradient of ``loss`` with respect to ``params`` based on a user-defined
        ``config``.

        :param loss: Outputs of the differentiated function.
        :type loss: Tensor
        :param params: Inputs with respect to which the gradient will be returned.
        :type params: Sequence of Tensor
        :param paths: Paths on which the gradient will be calculated.
        :type paths: List of list of Problem
        :param create_graph:
            If ``True``, graph of the derivative will be constructed, allowing to compute higher order
            derivative products. Default: ``True``.
        :type create_graph: bool, optional
        :param retain_graph:
            If ``False``, the graph used to compute the grad will be freed. Note that in nearly all
            cases setting this option to ``True`` is not needed and often can be worked around in a much
            more efficient way. Defaults to the value of ``create_graph``.
        :type retain_graph: bool, optional
        :param allow_unused:
            If ``False``, specifying inputs that were not used when computing outputs (and therefore
            their grad is always zero) is an error. Defaults to ``False``.
        :type allow_unused: bool, optional
        """
        # direct grad
        if len(paths) > 0 or not self.gradient_accumulation_boundary():
            grads = torch.autograd.grad(
                loss,
                params,
                create_graph=create_graph,
                retain_graph=retain_graph,
                allow_unused=allow_unused,
            )
            self.set_grads(params, grads)
        else:
            torch.autograd.backward(
                loss,
                inputs=params,
                create_graph=create_graph,
                retain_graph=retain_graph,
            )

        # indirect grad: best-response Jacobian
        if self._config.first_order:
            for idx, path in enumerate(paths):
                retain_graph_implicit = False if idx == len(paths) - 1 else True
                do_sync = bool(
                    idx == len(paths) - 1 and self.gradient_accumulation_boundary()
                )
                grads = get_grads(loss, path, retain_graph_implicit, do_sync)
                if not do_sync:
                    self.set_grads(params, grads)

    def set_grads(self, params, grads):
        """
        Set gradients for trainable parameters. ``params.grad = grads``

        :param params: Trainable parameters
        :type params: Sequence of Tensor
        :param grads: Calculated gradient
        :type grads: Sequence of Tensor
        """
        for param, grad in zip(params, grads):
            if grad is not None:
                if hasattr(param, "grad") and param.grad is not None:
                    param.grad = param.grad + grad
                else:
                    param.grad = grad

    def synchronize_params(self, params):
        """
        synchronize parameters across distributed data-parallel processes
        """
        if self._world_size > 1 and self._strategy not in ["fsdp", "accelerate"]:
            for param in params:
                dist.broadcast(param.data, 0)

    @abc.abstractmethod
    def optimizer_step(self, *args, **kwargs):
        """
        Update weights as in PyTorch's native ``optim.step()``
        """
        raise NotImplementedError

    def zero_grad(self):
        """
        Set gradients for trainable parameters for the current problem to 0.
        Similar with PyTorch's ``optim.zero_grad()`` or ``module.zero_grad()``.
        """
        for param in list(self.trainable_parameters()):
            if hasattr(param, "grad"):
                del param.grad

    def clip_grad(self):
        """
        Perform gradient clipping based on the norm provided by Config
        """
        if self._strategy != "fsdp":
            torch.nn.utils.clip_grad_norm_(
                parameters=self.trainable_parameters(), max_norm=self.gradient_clipping
            )
        else:
            self.module.clip_grad_norm_(max_norm=self.gradient_clipping)

    def state_dict(self):
        """
        Return all states involved in ``Problem`` with a Python dictionary. By default, it
        includes ``self.module.state_dict`` and ``self.optimizer.state_dict``. Depending on users'
        configurations, it may include ``self.scheuler.state_dict`` (lr scheduler) and
        ``self.scaler.state_dict`` (fp16 training)
        """
        state_dict = {}
        state_dict["module"] = self.module.state_dict()
        state_dict["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler.state_dict()
        if self._is_default_fp16():
            state_dict["scaler"] = self.scaler.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        """Load the state for the ``Problem``

        Args:
            state_dict (dict): Python dictionary of Problem states.
        """
        self.module.load_state_dict(state_dict["module"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.scheduler is not None and "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        if self._is_default_fp16() and "scaler" in state_dict:
            self.scaler.load_state_dict(state_dict["scaler"])

    def configure_distributed_training(self, dictionary):
        """
        Set the configuration for distributed training.

        :param dictionary: Python dictionary of distributed training provided by Engine.
        :type dictionary: dict
        """
        self._strategy = dictionary["strategy"]
        self._backend = dictionary["backend"]
        self._world_size = dictionary["world_size"]
        self._rank = dictionary["rank"]
        self._local_rank = dictionary["local_rank"]

    def configure_roll_back(self, roll_back):
        """
        Set the roll-back (warm- start) option from Engine

        :param roll_back: roll-back (warm-start) on/off
        :type roll_back: bool
        """
        if len(self._parents) > 0:
            self._roll_back = roll_back

    def configure_device(self):
        """
        Set the device for the current problem.
        """
        if self._strategy in ["distributed", "zero", "fsdp"]:
            torch.cuda.set_device(self._local_rank)
            self.device = torch.device("cuda", self._local_rank)
        elif self._strategy == "accelerate":
            self.device = self.accelerator.device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_opt_param_group_for_param(self, param):
        """
        Get optimizer param_group for specific parameter

        :param param: Parameter for which optimizer param_group is inquired
        :type param: torch.nn.Parameter
        :return: param_group for the given parameter
        :rtype: dict
        """
        param_groups = self.optimizer.param_groups
        for group in param_groups:
            for p in group["params"]:
                if param is p:
                    return group

    def get_opt_state_for_param(self, param):
        """
        Get optimizer state for specific parameter

        :param param: Parameter for which optimizer state is inquired
        :type param: torch.nn.Parameter
        :return: optimizer state for the given parameter
        :rtype: dict
        """
        state = self.optimizer.state
        return state[param]

    @abc.abstractmethod
    def cache_states(self):
        """
        Cache params, buffers, optimizer states when ``config.roll_back`` is set to ``True`` in
        ``step``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def recover_states(self):
        """
        Recover params, buffers, optimizer states when ``config.roll_back`` is set to ``True`` in
        ``step``.
        """
        raise NotImplementedError

    def on_epoch_end_exec(self):
        if self.is_implemented("on_epoch_end"):
            self.on_epoch_end()

    def gradient_accumulation_boundary(self):
        """
        Check whether the current step is on the gradient accumulation boundary
        """
        return bool(self._count % self.gas == 0)

    def _is_default_fp16(self):
        """
        Check whether to use PyTorch native fp16 (mixed-precision) feature
        """
        if not self._fp16 or self._strategy in ["accelerate"]:
            return False
        return True

    def is_implemented(self, fn_name):
        """
        Check if ``fn_name`` method is implemented in the class

        :rtype: bool
        """
        return callable(getattr(self, fn_name, None))

    def check_ready(self):
        """
        Check if unrolling processes of lower level problems in the hierarchical dependency
        graph are all ready/done. ``step`` function is only excuted when this method returns
        ``True``.

        :rtype: bool
        """
        return all(self.ready)

    def log(self, stats, global_step):
        """
        Log (training) stats to the ``self.logger``

        :param stats: log metrics such as loss and classification accuracy.
        :type stats: Any
        :param step: global/local step associated with the ``stats``.
        :type step: int
        """
        loss_log = log_from_loss_dict(stats)
        if global_step is None:
            self.logger.info(
                f'[Problem "{self._name}"] [Local Step {self._count}] {loss_log}'
            )
        else:
            self.logger.info(
                f'[Problem "{self._name}"] [Global Step {global_step}] [Local Step {self._count}] '
                f"{loss_log}"
            )
        cur_step = global_step
        if global_step is None or self.log_local_step:
            cur_step = self._count
        self.logger.log(stats, tag=self._name, step=cur_step)

    def add_child(self, problem):
        """
        Add ``problem`` to the lower-level problem list.

        :param problem: lower-level problem in the dependency graph
        :type problem: Problem
        """
        assert problem not in self._children
        self._children.append(problem)

    def add_parent(self, problem):
        """
        Add ``problem`` to the upper-level problem list.

        :param problem: upper-level problem in the dependency graph
        :type problem: Problem
        """
        assert problem not in self._parents
        self._parents.append(problem)

    def add_paths(self, paths):
        """
        Add new hypergradient backpropagation paths.
        """
        self._paths.extend(paths)

    def add_logger(self, logger):
        """
        Add logger to the current problem.

        :param logger: logger defined by users in ``Engine``.
        """
        if self.logger is None:
            self.logger = logger

    def add_env(self, env):
        """
        Add environment to the current problem.

        :param env: Environment.
        """
        if self.env is None:
            self.env = env

    @abc.abstractmethod
    def parameters(self):
        """
        Return all parameters for the current problem.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def trainable_parameters(self):
        """
        Define all *trainable* parameters for the current problem.
        """
        raise NotImplementedError

    def clear_dependencies(self):
        """
        Clear the dependencies of the current problem.
        """
        self._children = []
        self._parents = []
        self._paths = []

    def train(self):
        """
        Set the current problem to the training mode.
        """
        self._training = True

    def eval(self):
        """
        Set the current problem to the evaluation mode.
        """
        self._training = False

    def is_rank_zero(self):
        """
        Check whether the current device is rank 0.
        """
        return self._rank == 0

    @property
    def name(self):
        """[summary]
        Return the user-defined name of the module.
        """
        return self._name

    @property
    def config(self):
        """
        Return the configuration for the current problem.
        """
        return self._config

    @property
    def children(self):
        """
        Return lower-level problems for the current problem.
        """
        return self._children

    @property
    def parents(self):
        """
        Return upper-level problems for the current problem.
        """
        return self._parents

    @property
    def paths(self):
        """
        Return hypergradient calculation paths for the current problem.
        """
        return self._paths

    @property
    def leaf(self):
        """
        Return whether the current problem is leaf or not.

        :return: leaf
        :rtype: bool
        """
        return self._leaf

    @property
    def count(self):
        """
        Return the local step for the current problem.

        :return: local step
        :rtype: int
        """
        return self._count

    @leaf.setter
    def leaf(self, leaf):
        """
        Set the current problem as a leaf problem.
        """
        self._leaf = leaf
