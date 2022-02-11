from .maml import maml
from .implicit import implicit
from .neuman import neuman

grad_fn_mapping = {
    'maml': maml,
    'implicit': implicit,
    'neuman': neuman
}

def get_grad_fn(grad_fn_type):
    assert grad_fn_type in grad_fn_mapping
    return grad_fn_mapping[grad_fn_type]
