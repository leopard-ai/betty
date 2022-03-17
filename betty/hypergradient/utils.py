import torch

def concat(xs):
    return torch.cat([x.view(-1) for x in xs])

def sub_none(a, b):
    if a is None and b is None:
        raise ValueError
    if a is None:
        return -b
    elif b is None:
        return a
    else:
        return a - b
