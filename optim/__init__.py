import abc

import torch

from .adam import *
from .adamw import *
from .sgd import *


def patch_optimizer(optimizer, loss, data_loader):
    """[summary]
    Take non-differentiable optimizer (e.g., PyTorch's native optimizers),
    and return differentiable optimizer
    """
    return DifferentiableOptimizer(optimizer, loss, data_loader)


class DifferentiableOptimizer:
    """[summary]
    Make optimizer.step() differentiable
    """
    def __init__(self, optimizer, loss, data_loader):
        self.optimizer = optimizer
        self.loss = loss
        self.data_loader = iter(data_loader)

    def __call__(self, params, inners, outers, first_order=False):
        return self.step(params, inners, outers, first_order)

    def step(self, params, inners, outers, first_order=False):
        with torch.enable_grad():
            data = next(self.data_loader)
            loss = self.loss(data, params, inners, outers)
            grads = torch.autograd.grad(loss, params, create_graph=not first_order)
            for param, grad in zip(params, grads):
                param.grad = grad
            self._update(params, inners, outers)

    @abc.abstractmethod
    def _update(self, params, inners, outers):
        pass
