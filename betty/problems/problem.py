# Copyright Sang Keun Choe
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc

import torch

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
        config,
        module=None,
        optimizer=None,
        scheduler=None,
        train_data_loader=None,
        device=None,
    ):
        self._name = name
        self._config = config if config is not None else Config()
        self.engine_config = None
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # computation graph depedency
        # ! dependency can be defined both in ``Module'' class and ``Engine'' class
        self._parents = []
        self._children = []
        self._paths = []
        self._problem_name_dict = {}

        # data loader
        self.train_data_loader = train_data_loader
        self.train_data_iterator = None
        self.cur_batch = None

        # module
        self.module = module

        # optimizer & lr scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler

        # environment
        self.env = None

        # fp16
        self._fp16 = config.fp16
        if self._fp16:
            self.initial_dynamic_scale = config.initial_dynamic_scale
            self.scale_factor = config.scale_factor

        # gradient accumulation
        self.gas = config.gradient_accumulation

        # logger
        self.logger = None
        self.log_step = config.log_step
        self.log_local_step = config.log_local_step

        # misc
        self._leaf = False
        self._default_grad = False
        self._first_order = False
        self._retain_graph = config.retain_graph
        self._allow_unused = config.allow_unused
        self._unroll_steps = config.unroll_steps
        self._roll_back = False
        self._inner_loop_start = True
        self._training = True
        self.ready = None
        self._count = 0

    def initialize(self, engine_config):
        """
        ``initialize`` patches/sets up module, optimizer, data loader, etc. after compiling a
        user-provided configuration (e.g., fp16 training, iterative differentiation)
        """
        # engine config
        self.engine_config = engine_config

        # initialize update ready to False
        if self._leaf:
            assert len(self._children) == 0
        if len(self._paths) == 0:
            self._default_grad = True
        self.ready = [False for _ in range(len(self._children))]

        # compile parents configurations
        first_order = []
        for problem in self._parents:
            parent_config = problem.config
            first_order.append(parent_config.first_order)
        self._first_order = all(first_order)
        if len(self._parents) > 0:
            self._roll_back = engine_config.roll_back

        # compile children configurations
        children_unroll_steps = []
        for problem in self._children:
            child_config = problem.config
            children_unroll_steps.append(child_config.unroll_steps)
        if len(children_unroll_steps) > 0:
            assert all(s == children_unroll_steps[0] for s in children_unroll_steps)

        self._inner_loop_start = True

        # set up data loader
        train_data_loader = self.train_data_loader
        if self.is_implemented("configure_train_data_loader"):
            train_data_loader = self.configure_train_data_loader()
        if train_data_loader is not None:
            assert self.is_implemented("get_batch")
            self.train_data_iterator = iter(train_data_loader)

        # set up module for the current level
        if self.is_implemented("configure_module"):
            if self.configure_module() is not None:
                self.module = self.configure_module()
        assert self.module is not None, "Module must be specified!"
        self.module.to(self.device)
        if engine_config.distributed and torch.cuda.device_count() > 1:
            self.module = torch.nn.DataParallel(self.module)

        # set up optimizer
        if self.is_implemented("configure_optimizer"):
            if self.configure_optimizer() is not None:
                self.optimizer = self.configure_optimizer()

        # set up lr scheduler
        if self.is_implemented("configure_scheduler"):
            if self.configure_scheduler is not None:
                self.scheduler = self.configure_scheduler()

        # set up fp16 training
        if self._fp16:
            assert torch.cuda.is_available()
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=self.initial_dynamic_scale, growth_factor=self.scale_factor
            )

        # Logging INFO
        # TODO: Replace print with logging
        path_str = [[node.name for node in path] for path in self._paths]
        children_str = [node.name for node in self._children]
        parents_str = [node.name for node in self._parents]
        self.logger.info("*** Problem Information ***")
        self.logger.info(f"Name: {self._name}")
        self.logger.info(f"Uppers: {parents_str}")
        self.logger.info(f"Lowers: {children_str}")
        self.logger.info(f"Paths: {path_str}\n")

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
        if self._fp16:
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
            if self.log_step > 0 and self._count % self.log_step == 0:
                self.log(loss_dict, global_step)

            # call parent step_normal after unrolling
            if self._training and self._count % (self._unroll_steps * self.gas) == 0:
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
        self.step_normal(global_step=global_step)
        if self._count % (self._unroll_steps * self.gas) == 0:
            self.step_after_roll_back()

    def get_batch(self):
        """
        Load training batch from the user-provided data loader

        :return: New training batch
        :rtype: Any
        """
        try:
            batch = next(self.train_data_iterator)
        except StopIteration:
            train_data_loader = self.train_data_loader
            if self.is_implemented("configure_train_data_loader"):
                train_data_loader = self.configure_train_data_loader()
            self.train_data_iterator = iter(train_data_loader)
            batch = next(self.train_data_iterator)
        batch = tuple(convert_tensor(item, self.device, self._fp16) for item in batch)

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
        if self._fp16:
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
        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=create_graph,
            retain_graph=retain_graph,
            allow_unused=allow_unused,
        )
        self.set_grads(params, grads)

        # indirect grad: best-response Jacobian
        if self._config.first_order:
            for idx, path in enumerate(paths):
                retain_graph_implicit = False if idx == len(paths) - 1 else True
                grads = get_grads(loss, path, retain_graph_implicit)
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

    def state_dict(self):
        state_dict = {}
        state_dict["module"] = self.module.state_dict()
        state_dict["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler.state_dict()
        if self._fp16:
            state_dict["scaler"] = self.scaler.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict["module"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.scheduler is not None and "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        if self._fp16 and "scaler" in state_dict:
            self.scaler.load_state_dict(state_dict["scaler"])

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

    def set_problem_attr(self, problem):
        """
        Set class attributes for upper-/lower-level problems based on their names.

        :param problem: lower- or upper-level problem in the dependency graph
        :type problem: Problem
        :return: name of ``problem``
        :rtype: str
        """
        name = problem.name
        if name not in self._problem_name_dict:
            assert not hasattr(
                self, name
            ), f"Problem already has an attribute named {name}!"
            self._problem_name_dict[name] = 0
            setattr(self, name, problem)
        elif self._problem_name_dict[name] == 0:
            # rename first problem
            first_problem = getattr(self, name)
            delattr(self, name)
            setattr(self, name + "_0", first_problem)

            self._problem_name_dict[name] += 1
            name = name + "_" + str(self._problem_name_dict[name])
            setattr(self, name, problem)
        else:
            self._problem_name_dict[name] += 1
            name = name + "_" + str(self._problem_name_dict[name])
            setattr(self, name, problem)

        return name

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

        :param logger: logger defined by users in ``Engine``.
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
        Set the current problem as a leaf problem
        """
        self._leaf = leaf
