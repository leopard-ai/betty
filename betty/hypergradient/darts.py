import torch
import torch.distributed as dist

from betty.hypergradient.utils import grad, precondition
from betty.utils import to_vec, replace_none_with_zero


def darts(vector, curr, prev, sync):
    """
    Approximate the matrix-vector multiplication with the best response Jacobian by the
    finite difference method. More specifically, we modified the finite difference method proposed
    in `DARTS: Differentiable Architecture Search <https://arxiv.org/pdf/1806.09055.pdf>`_ by
    re-interpreting it from the implicit differentiation perspective. Empirically, this method
    achieves better memory efficiency, training wall time, and test accuracy that other methods.

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
    is_fsdp = True if curr._strategy == "fsdp" else False
    R = config.darts_alpha
    vector_norm = to_vec(vector).norm()
    if is_fsdp:
        vector_norm_sq = vector_norm.pow(2)
        dist.all_reduce(vector_norm_sq, op=dist.ReduceOp.SUM)
        vector_norm = vector_norm_sq.sqrt()
    eps = R / vector_norm.add_(1e-15).item()

    for p, v in zip(curr.meta_trainable_parameters(), vector):
        p.data.add_(v.data, alpha=eps)
    loss_p = curr.training_step_exec(curr.cur_batch)
    grad_p = grad(
        loss_p, prev.trainable_parameters(), allow_unused=True, is_fsdp=is_fsdp
    )
    grad_p = replace_none_with_zero(grad_p, prev.trainable_parameters())
    if sync:
        grad_p = [-g_p.div_(2 * eps) for g_p in grad_p]
        prev.set_grads(prev.trainable_parameters(), grad_p)

    # negative
    for p, v in zip(curr.meta_trainable_parameters(), vector):
        p.data.sub_(v.data, alpha=2 * eps)
    loss_n = curr.training_step_exec(curr.cur_batch)
    if sync:
        torch.autograd.backward(loss_n / (2 * eps), inputs=prev.trainable_parameters())
    else:
        grad_n = grad(
            loss_n, prev.trainable_parameters(), allow_unused=True, is_fsdp=is_fsdp
        )
        grad_n = replace_none_with_zero(grad_n, prev.trainable_parameters())

    # reverse weight change
    if not config.darts_multitask:
        for p, v in zip(curr.meta_trainable_parameters(), vector):
            p.data.add_(v.data, alpha=eps)

    implicit_grad = None
    if not sync:
        implicit_grad = [(x - y).div_(2 * eps) for x, y in zip(grad_n, grad_p)]

    return implicit_grad
