import torch

from .sgd import fsgd
from .adam import fadam
from .adamw import fadamw

optimizer_mapping = {
    torch.optim.SGD: fsgd,
    torch.optim.Adam: fadam,
    torch.optim.AdamW: fadamw
}

def get_update_fn(optimizer):
    """[summary]
    Return (functional) udpate function for the given optimizer (e.g., F.sgd, F.Adam)
    """
    assert type(optimizer) in optimizer_mapping

    return optimizer_mapping[type(optimizer)]
