import torch

from .sgd import PatchedSGD
from .adam import PatchedAdam
from .adamw import PatchedAdamW

optimizer_mapping = {
    torch.optim.SGD: PatchedSGD,
    torch.optim.Adam: PatchedAdam,
    torch.optim.AdamW: PatchedAdamW
}

def get_patched_optimizer(optimizer):
    """[summary]
    Return patched optimizer that allows gradient flow for parameter update given
    """
    assert type(optimizer) in optimizer_mapping

    patched_optimizer_cls = optimizer_mapping[type(optimizer)]
    patched_optimizer = patched_optimizer_cls(optimizer)
    return patched_optimizer
