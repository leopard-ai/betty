import abc
from dataclasses import dataclass

import torch
import higher

import optim


@dataclass
class HypergradientConfig:
    type: 'string' = 'implicit'
    step: int = 1
    first_order: bool = False
    leaf: bool = False


class Module:
    def __init__(self,
                 data_loader,
                 config,
                 device=None):
        self._config = config
        self.device = device

        self.data_loader = iter(data_loader)

        # ! Maybe users want to define parents and children in the same way they define modules for
        # ! the current level problem
        # ! One idea is to let them configure modules in each level using `configure_level` member
        # ! function
        self._parents = []
        self._children = []

        self.module = None
        self.fmodule = None
        self.optimizer = None

        self.ready = None
        self.count = 0
        self._first_order = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """[summary]
        Users define how forward call is defined for the current problem.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def training_step(self, batch, *args, **kwargs):
        """[summary]
        Users define how loss is calculated for the current problem.
        """
        # (batch, batch_idx)
        raise NotImplementedError

    def step(self, *args, **kwargs):
        """[summary]
        Perform gradient calculation and update parameters accordingly
        """
        if self.check_ready():
            self.count += 1

            batch = next(self.data_loader)
            loss = self.training_step(batch, *args, **kwargs)
            self.backward(loss, self.trainable_parameters(), self._first_order)
            new_params = self.optimizer.step()
            self.fmodule.update_params(new_params)

            for problem in self._parents:
                if self.count % problem.hgconfig().step == 0:
                    idx = problem.children().index(self)
                    problem[idx] = True
                    problem.step()

            self.ready = [False for _ in range(len(self._children))]

    def backward(self, loss, params, first_order=False):
        """[summary]
        Calculate and return gradient for given loss and parameters
        Args:
            loss ([type]): [description]
            params ([type]): [description]
            first_order (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        if self._config.leaf:
            grad = torch.autograd.grad(loss, params, create_graph=not first_order)
            for p, g in zip(params, grad):
                if hasattr(p, 'gradient') and p.gradient is not None:
                    p.gradient = p.gradient + g
                else:
                    p.gradient = g
        else:
            assert len(self._children) > 0
            if self._config.type == 'implicit':
                raise NotImplementedError
            elif self._config.type == 'maml':
                raise NotImplementedError

    def zero_grad(self):
        """[summary]
        Set gradients for trainable parameters for the current problem to 0.
        """
        for param in self.trainable_parameters():
            param.gradient = None

    def initialize(self):
        """[summary]
        Initialize basic things
        """
        self.ready = [False for _ in range(len(self._children))]

        first_order = []
        for problem in self._parents:
            hgconfig = problem.config()
            first_order.append(hgconfig.first_order)
        self._first_order = all(first_order)

        self.patch_models()
        self.patch_optimizer()

    def patch_optimizer(self):
        """[summary]
        Patch optimizer to avoid in-place operations so that gradient flows through param update.
        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError

    def patch_models(self):
        """[summary]
        Patch models to support functional forward that takes params as an input
        """
        self.fmodule = higher.monkeypatch(self.module,
                                          device=self.device,
                                          track_higher_grads=not self._first_order)

    def check_ready(self):
        """[summary]
        Check if parameter updates in all children are ready
        """
        if self._config.leaf:
            return True
        ready = all(self.ready)
        return ready

    def add_child(self, problem):
        """[summary]
        Add a new problem to the children node list.
        """
        assert problem not in self._children
        assert problem not in self._parents
        self._children.append(problem)

    def add_parent(self, problem):
        """[summary]
        Add a new problem to the parent node list.
        """
        assert problem not in self._children
        assert problem not in self._parents
        self._parents.append(problem)

    def parameters(self):
        """[summary]
        Return parameters for the current problem.
        """
        return self.fmodule.fast_params

    def trainable_parameters(self):
        """[summary]
        Return trainable parameters for the current problem.
        """
        return [p if p.requires_grad else torch.tensor([], requires_grad=True)
            for p in self.parameters()]

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
