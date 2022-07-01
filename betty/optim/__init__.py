import inspect

import torch

from .sgd import DifferentiableSGD
from .adam import DifferentiableAdam
from .adamw import DifferentiableAdamW

optimizer_mapping = {
    torch.optim.SGD: DifferentiableSGD,
    torch.optim.Adam: DifferentiableAdam,
    torch.optim.AdamW: DifferentiableAdamW,
}


def patch_optimizer(optimizer, module):
    """Patch PyTorch's native optimizer by replacing all in-place operations in its ``step`` method

    :param optimizer: User-provided PyTorch's native optimizer
    :type optimizer:
        `torch.nn.optim.Optimizer
        <https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer>`_
    :param module: User-provided PyTorch module that is being optimized by the ``optimizer``
    :type module:
        `torch.nn.Module
        <https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=module#torch.nn.Module>`_
    :return: Corresponding differentiable optimizer
    :rtype: DifferentiableOptimizer
    """
    assert type(optimizer) in optimizer_mapping

    return optimizer_mapping[type(optimizer)](optimizer, module)


def patch_scheduler(old_scheduler, new_optimizer):
    """Patch the original learning rate scheduler to work with a patched differentiable optimizer.

    :param old_scheduler: User-provided PyTorch's learning rate scheduler
    :type old_scheduler:
        `torch.optim.lr_scheduler
        <https://pytorch.org/docs/stable/optim.html?highlight=lr_scheduler>`_
    :param new_optimizer: Patched differentiable optimizer
    :type new_optimizer: DifferentiableOptimizer
    :return: Patched learning rate scheduler
    :rtype:
        `torch.optim.lr_scheduler
        <https://pytorch.org/docs/stable/optim.html?highlight=lr_scheduler>`_
    """
    kwargs = {}
    sig = inspect.signature(old_scheduler.__class__.__init__)
    for param in sig.parameters:
        key = param
        if key == "self":
            continue
        elif key == "optimizer":
            kwargs[key] = new_optimizer
        elif key == "last_epoch":
            kwargs[key] = getattr(old_scheduler, key) - 1
        else:
            value = getattr(old_scheduler, key)
            kwargs[key] = value
    new_scheduler = old_scheduler.__class__(**kwargs)

    return new_scheduler
