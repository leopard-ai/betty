import torch

from betty.utils import to_vec, replace_none_with_zero
from betty.hypergradient.utils import precondition


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
    R = config.darts_alpha
    if curr._strategy == "fsdp":
        curr_flat_param = curr.module._fsdp_wrapped_module.flat_param
        param_len = curr_flat_param.numel() - curr_flat_param._shard_numel_padded
        offset = curr._rank * param_len
        vector = [vector[0][offset : offset + param_len]]
    if config.darts_preconditioned and curr._strategy not in ["zero", "fsdp"]:
        vector = precondition(vector, curr)
    eps = R / to_vec(vector).norm().add_(1e-12).item()

    for p, v in zip(curr.trainable_parameters(), vector):
        p.data.add_(v.data, alpha=eps)
    loss_p = curr.training_step_exec(curr.cur_batch)
    grad_p = torch.autograd.grad(loss_p, prev.trainable_parameters(), allow_unused=True)
    grad_p = replace_none_with_zero(grad_p, prev.trainable_parameters())
    if sync:
        grad_p = [-g_p.div_(2 * eps) for g_p in grad_p]
        if prev._strategy == "fsdp":
            prev_flat_param = prev.module._fsdp_wrapped_module.flat_param
            offset = prev._rank * prev_flat_param.numel()
            valid_len = prev_flat_param.numel() - prev_flat_param._shard_numel_padded
            prev_grad_shard = grad_p[0].narrow(0, offset, valid_len)
            new_grad_p = torch.zeros_like(prev.trainable_parameters()[0])
            new_grad_p[:valid_len] = prev_grad_shard
            grad_p = [new_grad_p]
        prev.set_grads(prev.trainable_parameters(), grad_p)

    # negative
    for p, v in zip(curr.trainable_parameters(), vector):
        p.data.add_(v.data, alpha=-2 * eps)
    loss_n = curr.training_step_exec(curr.cur_batch)
    if sync:
        torch.autograd.backward(loss_n / (2 * eps), inputs=prev.trainable_parameters())
    else:
        grad_n = torch.autograd.grad(
            loss_n, prev.trainable_parameters(), allow_unused=True
        )
        grad_n = replace_none_with_zero(grad_n, prev.trainable_parameters())

    # reverse weight change
    for p, v in zip(curr.trainable_parameters(), vector):
        p.data.add(v.data, alpha=eps)

    implicit_grad = None
    if not sync:
        implicit_grad = [(x - y).div_(2 * eps) for x, y in zip(grad_n, grad_p)]

    return implicit_grad
