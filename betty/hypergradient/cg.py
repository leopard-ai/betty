import torch

from betty.hypergradient.utils import sub_with_none

def cg(loss, params, child, create_graph=True, retain_graph=False, allow_unused=True):
    raise NotImplementedError
