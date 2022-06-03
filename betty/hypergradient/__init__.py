import torch

from .darts import darts
from .cg import cg
from .neumann import neumann
from .reinforce import reinforce


jvp_fn_mapping = {"darts": darts, "neumann": neumann, "cg": cg, "reinforce": reinforce}


def get_grads(loss, path):
    jvp = torch.autograd.grad(loss, path[1].trainable_parameters())
    for i in range(1, len(path) - 1):
        jvp_fn_type = path[i].config.type
        assert jvp_fn_type in jvp_fn_mapping
        jvp_fn = jvp_fn_mapping[jvp_fn_type]
        jvp = jvp_fn(jvp, path[i], path[i + 1])

    return jvp
