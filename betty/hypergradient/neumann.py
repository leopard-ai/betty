import warnings

import torch

from betty.utils import neg_with_none


def neumann(vector, curr, prev, sync):
    """
    Approximate the matrix-vector multiplication with the best response Jacobian by the
    Neumann Series as proposed in
    `Optimizing Millions of Hyperparameters by Implicit Differentiation
    <https://arxiv.org/abs/1911.02590>`_ based on implicit function theorem (IFT). Users may
    specify learning rate (``neumann_alpha``) and unrolling steps (``neumann_iterations``) in
    ``Config``.

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
    # ! Mabye replace with child.loss by adding self.loss attribute to save computation
    assert len(curr.paths) == 0, "neumann method is not supported for higher order MLO!"
    config = curr.config
    in_loss = curr.training_step_exec(curr.cur_batch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        in_grad = torch.autograd.grad(
            in_loss, curr.trainable_parameters(), create_graph=True
        )
    v2 = approx_inverse_hvp(
        vector,
        in_grad,
        curr.trainable_parameters(),
        iterations=config.neumann_iterations,
        alpha=config.neumann_alpha,
    )
    if sync:
        v2 = [neg_with_none(x) for x in v2]
        torch.autograd.backward(
            in_grad, inputs=prev.trainable_parameters(), grad_tensors=v2
        )
        implicit_grad = None
    else:
        implicit_grad = torch.autograd.grad(
            in_grad, prev.trainable_parameters(), grad_outputs=v2
        )
        implicit_grad = [neg_with_none(ig) for ig in implicit_grad]

    return implicit_grad


def approx_inverse_hvp(v, f, params, iterations=3, alpha=1.0):
    p = v
    for _ in range(iterations):
        hvp = torch.autograd.grad(f, params, grad_outputs=v, retain_graph=True)
        v = [v_i - alpha * hvp_i for v_i, hvp_i in zip(v, hvp)]
        p = [v_i + p_i for v_i, p_i in zip(v, p)]

    return [alpha * p_i for p_i in p]
