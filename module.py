import typing as _typing
import abc
from dataclasses import dataclass
from collections import OrderedDict

import torch

import hypergrad as hg
import optim
import utils


@dataclass
class HypergradientConfig:
    type: 'string' = 'reverse'
    step: int = 1
    first_order: bool = False
    leaf: bool = False


class Problem:
    """[summary]
    """
    def __init__(self,
                 module: torch.nn.Module = None,
                 optimizer: torch.optim.Optimizer = None,
                 train_dataloader: _typing.Any = None,
                 valid_dataloader: _typing.Any = None,
                 hgconfig: 'HypergradientConfig' = None,
                 max_len: int = 1) -> None:
        # * problem
        self.module = module
        self.optimizer = self.configure_optimizer(optimizer)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        # * Hypergradient
        # Example: hypergradient = {'type': 'reverse', 'step': 3, 'leaf': True}
        self._hgconfig = hgconfig
        self._first_order = None
        self._param_history = utils.LimitedList(max_len)
        # prev: (problem, count)
        self._prevs = OrderedDict()
        # next: (problem, ready)
        self._nexts = OrderedDict()

    def forward(self, *args, **kwargs):
        """[summary]
        User defines how the forward function for the problem is defined.
        """
        self.module.forward(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abc.abstractmethod
    def training_step(self, batch):
        """[summary]
        User defines how training loss is defined.
        """
        # ! Currently, users need to call dependent problems with self._nexts[i] or self._prevs[i].
        # ! How to build a more straightforward and clean way to access other problems
        raise NotImplementedError

    @abc.abstractmethod
    def validation_step(self, batch):
        """[summary]
        User defines how validation loss is defined.
        """
        raise NotImplementedError

    def configure_optimizer(self, optimizer):
        """[summary]
        PyTorch's native optimizer updates model parameters in-place. This blocks users from
        calculating higher-order gradient, so we need to patch optimizer to avoid this issue.
        """
        return optim.patch_optimizer(optimizer, self.training_step, self.train_dataloader)

    def initialize(self):
        # * Check if first-order approximation will be used
        first_order = []
        for problem in self._prevs:
            hgconfig = problem.hgconfig()
            first_order.append(hgconfig.first_order)
        self._first_order = all(first_order)

    def step(self):
        # * perform current level step
        if self.check_inner_ready():
            self.optimizer.step(self.parameters(),
                                list(self._nexts.keys()),
                                list(self._prevs.keys()))
            for problem in self._prevs:
                self._prevs[problem] += 1
                if self._prevs[problem] % problem.hgconfig().step == 0:
                    problem.nexts()[self] = True
                    problem.step()

            for problem in self._nexts:
                self._nexts[problem] = False

    def set_param(self, params):
        """[summary]
        Set module's parameters to new parameters
        """
        raise NotImplementedError

    def calculate_gradients(self, loss, params, first_order=False):
        """[summary]
        Return gradient for loss with respect to parameters
        """
        if self._hgconfig.leaf:
            return torch.autograd.grad(loss, params, create_graph=not first_order)
        else:
            # TODO: Hypergradient calculation
            raise NotImplementedError

    def clone(self):
        """[summary]
        Return the copy of module
        """
        raise NotImplementedError

    def parameters(self):
        """[summary]
        Return module's parameters
        """
        return self.module.parameters()

    def check_inner_ready(self):
        """[summary]
        Check if parameter updates in all children are ready
        """
        if self._hgconfig.leaf:
            return True
        ready = all(list(self._nexts.values()))
        return ready

    def append_next(self, problem):
        assert problem not in self._nexts
        assert problem not in self._prevs
        self._nexts[problem] = False

    def append_prev(self, problem):
        assert problem not in self._nexts
        assert problem not in self._prevs
        self._prevs[problem] = 0

    @property
    def module(self):
        """[summary]
        Return module
        """
        return self.module

    @module.setter
    def module(self, module):
        """[summary]
        Set new module
        """
        self.module = module

    @property
    def optimizer(self):
        """[summary]
        Return optimizer
        """
        return self.optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """[summary]
        Set new optimizer
        """
        self.optimizer = self.configure_optimizer(optimizer)

    @property
    def train_dataloader(self):
        """[summary]
        Return train data loader
        """
        return self.train_dataloader

    @train_dataloader.setter
    def train_dataloader(self, loader):
        self.train_dataloader = loader

    @property
    def valid_dataloader(self):
        """[summary]
        Return valid data loader
        """
        return self.valid_dataloader

    @valid_dataloader.setter
    def valid_dataloader(self, loader):
        self.valid_dataloader = loader

    @property
    def hgconfig(self):
        """[summary]
        Return hypergradient configuration
        """
        return self._hgconfig

    @property
    def nexts(self):
        return self._nexts

    @property
    def prevs(self):
        return self._prevs
