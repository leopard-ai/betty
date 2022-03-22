import torch

def to_vec(tensor_list, alpha=1.):
    return torch.cat([alpha * t.reshape(-1) for t in tensor_list])

def sub_with_none(a, b):
    if a is None and b is None:
        raise ValueError
    if a is None:
        return -b
    elif b is None:
        return a
    else:
        return a - b
