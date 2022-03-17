from .maml import maml
from .darts import darts
from .neumann import neumann
from .cg import cg

grad_fn_mapping = {
    'maml': maml,
    'darts': darts,
    'neumann': neumann,
    'cg': cg
}

def get_grad_fn(grad_fn_type):
    assert grad_fn_type in grad_fn_mapping
    return grad_fn_mapping[grad_fn_type]
