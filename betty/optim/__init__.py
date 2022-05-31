import torch

from .sgd import DifferentiableSGD
from .adam import DifferentiableAdam
from .adamw import DifferentiableAdamW
from .lr_scheduler import patch_scheduler
from .fp16.fp16_optimizer import FP16_Optimizer

optimizer_mapping = {
    torch.optim.SGD: DifferentiableSGD,
    torch.optim.Adam: DifferentiableAdam,
    torch.optim.AdamW: DifferentiableAdamW
}

def patch_optimizer(optimizer, module):
    """[summary]
    Return differentiable optimizer for the given optimizer
    """
    assert type(optimizer) in optimizer_mapping

    return optimizer_mapping[type(optimizer)](optimizer, module)
