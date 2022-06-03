import torch

from betty.utils import add_with_none, neg_with_none


def reinforce(vector, curr, prev):
    """
    Approximate the matrix-vector multiplication with the best response Jacobian by the
    REINFORCE method. The use of REINFORCE algorithm allows users to differentiate through
    optimization with non-differentiable processes such as sampling. This method has not been
    completely implemented yet.
    """
    config = curr.config
