import torch

def to_vec(tensor_list, alpha=1.):
    return torch.cat([alpha * t.reshape(-1) for t in tensor_list])

def add_with_none(a, b):
    if a is None and b is None:
        return 0
    if a is None:
        return b
    elif b is None:
        return a
    else:
        return a + b

def neg_with_none(a):
    if a is None:
        return None
    else:
        return -a
