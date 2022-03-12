import abc
from collections.abc import Iterable
import copy
import typing
from dataclasses import dataclass

import torch
import functorch

import betty.optim as optim
import betty.hypergradient as hypergradient


@dataclass
class HypergradientConfig:
    type: str = 'maml'
    step: int = 2
    first_order: bool = False
    retain_graph: bool = False
    allow_unused: bool = False


class Module:
    def __init__(self,
                 name,
                 config,
                 module=None,
                 optimizer=None,
                 scheduler=None,
                 train_data_loader=None,
                 device=None):
        self._name = name
        self._config = config
        self.device = device

        # computation graph depedency
        # ! dependency can be defined both in ``Module'' class and ``Engine'' class
        self._parents = []
        self._children = []
        self._problem_name_dict = {}
        self.ready = None
        self.count = 0

        # data loader
        self.train_data_loader = train_data_loader
        self.train_data_iterator = None
        self.cur_batch = None

        # module
        self.module = module
        self.fmodule = None
        self.params = None
        self.buffers = None

        # optimizer & lr scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler

        # temp
        self.params_temp = None
        self.buffers_temp = None
        self.optimizer_state_temp = None
        self.grad_temp = None

        # misc
        self._leaf = False
        self._first_order = False
        self._retain_graph = config.retain_graph
        self._allow_unused = config.allow_unused
        self._inner_loop_start = True
        self._training = True

    def initialize(self):
        """[summary]
        Initialize basic things
        """
        # initialize update ready to False
        if self._leaf:
            assert len(self._children) == 0
        self.ready = [False for _ in range(len(self._children))]

        # initialize whether to track higher-order gradient for parameter update
        first_order = []
        for problem in self._parents:
            hgconfig = problem.config
            first_order.append(hgconfig.first_order)
        self._first_order = all(first_order)

        self._inner_loop_start = True

        # set up data loader
        train_data_loader = copy.deepcopy(self.train_data_loader)
        if self.is_implemented('configure_train_data_loader'):
            train_data_loader = self.configure_train_data_loader()
        assert train_data_loader is not None, "Train data loader must be specified!"
        self.train_data_iterator = iter(train_data_loader)

        # set up module for the current level
        if self.is_implemented('configure_module'):
            if self.configure_module() is not None:
                self.module = self.configure_module()
        assert self.module is not None, "Module must be specified!"

        # set up optimizer
        if self.is_implemented('configure_optimizer'):
            if self.configure_optimizer() is not None:
                self.optimizer = self.configure_optimizer()

        # set up lr scheduler
        if self.is_implemented('configure_scheduler'):
            if self.configure_scheduler is not None:
                self.scheduler = self.configure_scheduler()

        # patch model, optimizer, lr_scheduler to follow functional programming paradigm
        self.initialize_optimizer_state()
        self.patch_models()
        self.patch_optimizer()
        self.patch_scheduler()
        self.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """[summary]
        Users define how forward call is defined for the current problem.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def training_step(self, batch):
        """[summary]
        Users define how loss is calculated for the current problem.
        """
        raise NotImplementedError

    def step(self,
             backpropagate=True,
             param_update=True):
        """[summary]
        Perform gradient calculation and update parameters accordingly
        """
        if self.check_ready():
            if self._inner_loop_start:
                self.on_inner_loop_start()
                self._inner_loop_start = False
            if self._training and backpropagate:
                self.count += 1
            if not param_update:
                self.params_temp = copy.deepcopy(self.params)
                self.buffers_temp = copy.deepcopy(self.buffers)
                # TODO: replace with state_dict, load_state_dict
                self.optimizer_state_temp = copy.deepcopy(self.optimizer.state)

            # load data
            try:
                batch = next(self.train_data_iterator)
            except StopIteration:
                train_data_loader = copy.deepcopy(self.train_data_loader)
                if self.is_implemented('configure_train_data_loader'):
                    train_data_loader = self.configure_train_data_loader()
                self.train_data_iterator = iter(train_data_loader)
                batch = next(self.train_data_iterator)
            self.cur_batch = batch

            # calculate loss
            losses = self.training_step(self.cur_batch)
            if not (isinstance(losses, tuple) or isinstance(losses, list)):
                losses = (losses,)
            # TODO: Add custom loss aggregation
            # aggregate loss
            losses = tuple(loss / len(losses) for loss in losses)

            # calculate gradient (a.k.a backward)
            if len(losses) == 1:
                child = None if len(self._children) == 0 else self._children[0]
                grad = self.backward(loss=losses[0],
                                     params=self.params,
                                     child=child,
                                     create_graph=not self._first_order,
                                     retain_graph=self._retain_graph,
                                     allow_unused=self._allow_unused)
            else:
                assert len(losses) == len(self._children)
                for loss, child in zip(losses, self._children):
                    grad = self.backward(loss=loss,
                                         params=self.params,
                                         child=child,
                                         create_graph=not self._first_order,
                                         retain_graph=self._retain_graph,
                                         allow_unused=self._allow_unused)

            # calculate parameter update
            new_params = self.optimizer_step()
            self.params = new_params

            # zero-out grad
            self.zero_grad()

            # call parent step function
            if self._training and backpropagate:
                for problem in self._parents:
                    if self.count % problem.config.step == 0:
                        idx = problem.children.index(self)
                        problem.ready[idx] = True
                        problem.step()

                        self._inner_loop_start = True

                if not param_update:
                    self.params, self.params_temp = self.params_temp, None
                    self.buffers, self.buffers_temp = self.buffers_temp, None
                    # TODO: replace with state_dict, load_state_dict
                    self.optimizer.state = self.optimizer_state_temp
                    self.optimizer_state_temp = None

                    losses = self.training_step(self.cur_batch)
                    if not (isinstance(losses, tuple) or isinstance(losses, list)):
                        losses = (losses,)
                    losses = tuple(loss / len(losses) for loss in losses)

                    if len(losses) == 1:
                        child = None if len(self._children) == 0 else self._children[0]
                        grad = self.backward(loss=losses[0],
                                             params=self.params,
                                             child=child,
                                             create_graph=not self._first_order,
                                             retain_graph=self._retain_graph,
                                             allow_unused=self._allow_unused)
                    else:
                        assert len(losses) == len(self._children)
                        for loss, child in zip(losses, self._children):
                            grad = self.backward(loss=loss,
                                                 params=self.params,
                                                 child=child,
                                                 create_graph=not self._first_order,
                                                 retain_graph=self._retain_graph,
                                                 allow_unused=self._allow_unused)
                    grad_temp = None

                    new_params = self.optimizer_step()
                    self.params = new_params

                    self.zero_grad()

            self.ready = [False for _ in range(len(self._children))]

    def backward(self,
                 loss,
                 params,
                 child=None,
                 create_graph=True,
                 retain_graph=False,
                 allow_unused=False):
        """[summary]
        Calculate and return gradient for given loss and parameters
        Args:
            loss ([type]): [description]
            params ([type]): [description]
            first_order (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        if self.optimizer is None:
            return None

        if self._leaf or self.config.type == 'torch':
            grad = torch.autograd.grad(loss, params,
                                       create_graph=create_graph,
                                       retain_graph=retain_graph,
                                       allow_unused=allow_unused)
        else:
            assert len(self._children) > 0
            grad_fn = hypergradient.get_grad_fn(self.config.type)
            grad = grad_fn(loss, params, child,
                           create_graph=create_graph,
                           retain_graph=retain_graph,
                           allow_unused=allow_unused)

        # set gradient for each parameter
        for p, g in zip(params, grad):
            if hasattr(p, 'gradient') and p.gradient is not None:
                p.gradient = p.gradient + g
            else:
                p.gradient = g

        return grad

    def optimizer_step(self, *args, **kwargs):
        """[summary]
        Update weights as in native PyTorch's optim.step()
        """
        if self.optimizer is None:
            new_params = self.custom_optimizer_step(*args, **kwargs)
        else:
            new_params = self.optimizer.step(self.params)

        self.param_callback(new_params)

        return new_params

    def custom_optimizer_step(self):
        """[summary]
        Users define how optimizer step is performed. This is mainly used for developing
        meta- (or learnable) optimizer
        """
        return self.params

    def zero_grad(self):
        """[summary]
        Set gradients for trainable parameters for the current problem to 0.
        """
        for param in list(self.params):
            if hasattr(param, 'gradient'):
                del param.gradient
            if hasattr(param, 'grad'):
                del param.grad

    def grad_callback(self, grads):
        pass

    def param_callback(self, params):
        pass

    def on_inner_loop_start(self):
        pass

    def is_implemented(self, fn_name):
        return callable(getattr(self, fn_name, None))

    def initialize_optimizer_state(self):
        """[summary]
        Initialize optimizer state
        """
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                param.grad = torch.zeros_like(param.data)
        self.optimizer.step()

    def patch_models(self):
        """[summary]
        Patch models to support functional forward that takes params as an input
        """
        fmodule, params, buffers = functorch.make_functional_with_buffers(self.module)
        self.fmodule = fmodule
        self.params = params
        self.buffers = buffers

    def patch_optimizer(self):
        """[summary]
        Patch optimizer to avoid in-place operations so that gradient flows through param update.
        Raises:
            NotImplementedError: [description]
        """
        if self.optimizer is not None:
            self.optimizer = optim.patch_optimizer(self.optimizer, self.module)

    def patch_scheduler(self):
        """[summary]
        Patch scheduler to work on patched optimizer
        """
        if self.scheduler is not None:
            self.scheduler = optim.patch_scheduler(self.scheduler, self.optimizer)

    def check_ready(self):
        """[summary]
        Check if parameter updates in all children are ready
        """
        return all(self.ready)

    def set_problem_attr(self, problem):
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

    def add_child(self, problem):
        """[summary]
        Add a new problem to the children node list.
        """
        assert problem not in self._children
        assert problem not in self._parents
        self.set_problem_attr(problem)
        self._children.append(problem)

    def add_parent(self, problem):
        """[summary]
        Add a new problem to the parent node list.
        """
        assert problem not in self._children
        assert problem not in self._parents
        self.set_problem_attr(problem)
        self._parents.append(problem)

    def parameters(self):
        """[summary]
        Return parameters for the current problem.
        """
        return self.params

    def set_leaf(self):
        self._leaf = True

    def train(self):
        self._training = True

    def eval(self):
        self._training = False

    @property
    def name(self):
        return self._name

    @property
    def config(self):
        """[summary]
        Return the hypergradient configuration for the current problem.
        """
        return self._config

    @property
    def children(self):
        """[summary]
        Return children problems for the current problem.
        """
        return self._children

    @property
    def parents(self):
        """[summary]
        Return parent problemss for the current problem.
        """
        return self._parents
