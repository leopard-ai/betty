import abc

import torch


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
