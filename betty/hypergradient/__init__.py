from .maml import maml
from .darts import darts
from .neuman import neuman

grad_fn_mapping = {
    'maml': maml,
    'darts': darts,
    'neuman': neuman
}

def get_grad_fn(grad_fn_type):
    assert grad_fn_type in grad_fn_mapping
    return grad_fn_mapping[grad_fn_type]
