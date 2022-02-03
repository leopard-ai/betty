import abc
from dataclasses import dataclass

import torch
import higher

import optim


@dataclass
class HypergradientConfig:
    type: 'string' = 'reverse'
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
        self._currents = []
        self._parents = []
        self._children = []

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
    def training_step(self, *args, **kwargs):
        """[summary]
        Users define how loss is calculated for the current problem.
        """
        # (batch, batch_idx)
        raise NotImplementedError

    def step(self, *args, **kwargs):
        if self.check_ready():
            self.count += 1

            batch = next(self.data_loader)
            loss = self.training_step(batch)
            grads = self.backward(loss, self.parameters())
            # param.grad
            new_params = self.optimizer.step(grads)
            for module in self._currents:
                module.update_params(new_params)

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
        grads = {}
        if self._config.leaf:
            grad = torch.autograd.grad(loss, params, create_graph=not first_order)
            for p, g in zip(params, grad):
                grads[p] = g
        else:
            assert len(self._children) > 0
            if self._config.type == 'implicit':
                raise NotImplementedError
            elif self._config.type == 'maml':
                raise NotImplementedError 

        return grads

    def initialize(self):
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
        for idx, module in enumerate(self._currents):
            patched_module = higher.monkeypatch(module,
                                                device=self.device,
                                                track_higher_grads=not self._first_order)
            self._currents[idx] = patched_module

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
        Return (trainable) parameters for the current problem.
        """
        params = [module.parameters() for module in self._currents]
        return params

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
