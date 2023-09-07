import torch

from betty.utils import to_vec, replace_none_with_zero
from betty.hypergradient.utils import precondition


def sama(vector, curr, prev, sync):
    """
    Scalable Meta Learning Algorithm proposed in TBD.

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
    R = config.sama_adam_alpha

    vector = precondition(vector, curr)
    vector_norm = to_vec(vector).norm()
    eps = R / vector_norm.add_(1e-15).item()

    for p, v in zip(curr.meta_trainable_parameters(), vector):
        p.data.add_(v.data, alpha=eps)
    loss_p = curr.training_step_exec(curr.cur_batch)
    grad_p = torch.autograd.grad(loss_p, prev.trainable_parameters(), allow_unused=True)
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
        grad_n = torch.autograd.grad(
            loss_n, prev.trainable_parameters(), allow_unused=True
        )
        grad_n = replace_none_with_zero(grad_n, prev.trainable_parameters())

    # reverse weight change
    if not config.sama_multitask:
        for p, v in zip(curr.meta_trainable_parameters(), vector):
            p.data.add_(v.data, alpha=eps)
    else:
        curr.synchronize_params(curr.meta_trainable_parameters(), all_reduce=True)

    implicit_grad = None
    if not sync:
        implicit_grad = [(x - y).div_(2 * eps) for x, y in zip(grad_n, grad_p)]

    return implicit_grad
