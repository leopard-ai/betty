import torch

from betty.utils import replace_none_with_zero

from .darts import darts
from .sama import sama
from .cg import cg
from .neumann import neumann
from .reinforce import reinforce
from .utils import grad


jvp_fn_mapping = {
    "darts": darts,
    "sama": sama,
    "neumann": neumann,
    "cg": cg,
    "reinforce": reinforce,
}


def get_grads(loss, path, retain_graph, do_sync):
    is_fsdp = True if path[0]._strategy == "fsdp" else False
    jvp = grad(
        loss,
        path[1].meta_trainable_parameters(),
        retain_graph=retain_graph,
        allow_unused=True,
        is_fsdp=is_fsdp,
    )
    jvp = replace_none_with_zero(jvp, path[1].meta_trainable_parameters())
    for i in range(1, len(path) - 1):
        jvp_fn_type = path[i].config.type
        assert jvp_fn_type in jvp_fn_mapping
        jvp_fn = jvp_fn_mapping[jvp_fn_type]
        sync = bool(do_sync and i == len(path) - 2)
        jvp = jvp_fn(jvp, path[i], path[i + 1], sync)

    return jvp
