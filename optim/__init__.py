import torch

from .sgd import PatchedSGD
from .adam import PatchedAdam
from .adamw import PatchedAdamW

optimizer_mapping = {
    torch.optim.SGD: PatchedSGD,
    torch.optim.Adam: PatchedAdam,
    torch.optim.AdamW: PatchedAdamW
}

def get_update_fn(optimizer):
    """[summary]
    Return (functional) udpate function for the given optimizer (e.g., F.sgd, F.Adam)
    """
    assert type(optimizer) in optimizer_mapping

    return optimizer_mapping[type(optimizer)]
