import abc

import torch

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
    def __init__(self, original_optimizer):
        self.original_optimizer = original_optimizer

    @abc.abstractmethod
    def _update(self):
        pass

    def step(self):
        pass
