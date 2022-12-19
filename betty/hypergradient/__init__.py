import torch

from betty.utils import replace_none_with_zero

from .darts import darts
from .cg import cg
from .neumann import neumann
from .reinforce import reinforce


jvp_fn_mapping = {
    "darts": darts,
    "neumann": neumann,
    "cg": cg,
    "reinforce": reinforce,
}


def get_grads(loss, path, retain_graph, do_sync):
    jvp = torch.autograd.grad(
        loss,
        path[1].trainable_parameters(),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    jvp = replace_none_with_zero(jvp, path[1].trainable_parameters())
    for i in range(1, len(path) - 1):
        jvp_fn_type = path[i].config.type
        assert jvp_fn_type in jvp_fn_mapping
        jvp_fn = jvp_fn_mapping[jvp_fn_type]
        sync = bool(do_sync and i == len(path) - 2)
        jvp = jvp_fn(jvp, path[i], path[i + 1], sync)

    return jvp
