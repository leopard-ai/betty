import abc

import torch

from .adam import *
from .adamw import *
from .sgd import *


def patch_optimizer(optimizer):
    """[summary]
    Take non-differentiable optimizer (e.g., PyTorch's native optimizers),
    and return differentiable optimizer
    """
    return PatchedOptimizer(optimizer)


class PatchedOptimizer:
    """[summary]
    Make optimizer.step() differentiable
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer

    @abc.abstractmethod
    def _update(self, params, inners, outers):
        pass

    def step(self, grads):
        pass
