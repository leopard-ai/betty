import torch

def concat(xs):
    return torch.cat([x.view(-1) for x in xs])