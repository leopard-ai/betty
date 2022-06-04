import torch

from betty.utils import neg_with_none


def reinforce(vector, curr, prev):
    """
    Approximate the matrix-vector multiplication with the best response Jacobian by the
    REINFORCE method. The use of REINFORCE algorithm allows users to differentiate through
    optimization with non-differentiable processes such as sampling. This method has not been
    completely implemented yet.

    :param vector:
        Vector with which matrix-vector multiplication with best-response Jacobian (matrix) would
        be performed.
    :type vector: Sequence of Tensor
    :param curr: A current level problem
    :type curr: Problem
    :param prev: A directly lower-level problem to the current problem
    :type prev: Problem
    :return: (Intermediate) gradient
    :rtype: Sequence of Tensor
    """
    config = curr.config
