import warnings

import torch

from betty.utils import neg_with_none, to_vec


def cg(vector, curr, prev, sync):
    """
    Approximate the matrix-vector multiplication with the best response Jacobian by the
    (PyTorch's) default autograd method. Users may need to specify learning rate (``cg_alpha``) and
    conjugate gradient descent iterations (``cg_iterations``) in ``Config``.

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
    assert len(curr.paths) == 0, "cg method is not supported for higher order MLO!"
    config = curr.config
    in_loss = curr.training_step_exec(curr.cur_batch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        in_grad = torch.autograd.grad(
            in_loss, curr.trainable_parameters(), create_graph=True
        )

    x = [torch.zeros_like(vi) for vi in vector]
    r = [torch.zeros_like(vi).copy_(vi) for vi in vector]
    p = [torch.zeros_like(rr).copy_(rr) for rr in r]

    for _ in range(config.cg_iterations):
        hvp = torch.autograd.grad(
            in_grad, curr.parameters(), grad_outputs=p, retain_graph=True
        )
        hvp_vec = to_vec(hvp, alpha=config.cg_alpha)
        r_vec = to_vec(r)
        p_vec = to_vec(p)
        numerator = torch.dot(r_vec, r_vec)
        denominator = torch.dot(hvp_vec, p_vec)
        alpha = numerator / denominator

        x_new = [xx + alpha * pp for xx, pp in zip(x, p)]
        r_new = [rr - alpha * pp for rr, pp in zip(r, hvp)]
        r_new_vec = to_vec(r_new)
        beta = torch.dot(r_new_vec, r_new_vec) / numerator
        p_new = [rr + beta * pp for rr, pp in zip(r, p)]

        x, p, r = x_new, p_new, r_new
    x = [config.cg_alpha * xx for xx in x]

    if sync:
        x = [neg_with_none(x_i) for x_i in x]
        torch.autograd.backward(
            in_grad, inputs=prev.trainable_parameters(), grad_tensors=x
        )
        implicit_grad = None
    else:
        implicit_grad = torch.autograd.grad(
            in_grad, prev.trainable_parameters(), grad_outputs=x
        )
        implicit_grad = [neg_with_none(ig) for ig in implicit_grad]

    return implicit_grad
