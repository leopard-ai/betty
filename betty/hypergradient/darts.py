import torch

from betty.hypergradient.utils import to_vec, add_with_none


def darts(loss, params, path, config, create_graph=True, retain_graph=False, allow_unused=True):
    """Approximate the matrix-vector multiplication with the best response Jacobian by the
    finite difference method. More specifically, we modified the finite difference method proposed
    in `DARTS: Differentiable Architecture Search <https://arxiv.org/pdf/1806.09055.pdf>`_ by
    re-interpreting it from the implicit differentiation perspective. Empirically, this method
    achieves better memory efficiency, training wall time, and test accuracy that other methods.

    :param loss: Outputs of the differentiated function.
    :type loss: Tensor
    :param params: Inputs with respect to which the gradient will be returned.
    :type params: Sequence of Tensor
    :param path: Path on which the gradient will be calculated.
    :type path: List of Problem
    :param config: Hyperparameters for the best-response Jacobian approximation
    :type config: Config
    :param create_graph:
        If ``True``, graph of the derivative will be constructed, allowing to compute higher order
        derivative products. Default: ``True``.
    :type create_graph: bool, optional
    :param retain_graph:
        If ``False``, the graph used to compute the grad will be freed. Note that in nearly all
        cases setting this option to ``True`` is not needed and often can be worked around in a much
        more efficient way. Defaults to the value of ``create_graph``.
    :type retain_graph: bool, optional
    :param allow_unused:
        If ``False``, specifying inputs that were not used when computing outputs (and therefore
        their grad is always zero) is an error. Defaults to ``False``.
    :type allow_unused: bool, optional
    :return: The gradient of ``loss`` with respect to ``params``
    :rtype: List of Tensor
    """
    # direct grad
    direct_grad = torch.autograd.grad(loss,
                                      params,
                                      create_graph=create_graph,
                                      retain_graph=retain_graph,
                                      allow_unused=allow_unused)

    # implicit grad
    implicit_grad = torch.autograd.grad(loss, path[1].trainable_parameters())
    for i in range(1, len(path)-1):
        implicit_grad = darts_helper(implicit_grad, path[i], path[i+1], config)

    return [add_with_none(dg, ig) for dg, ig in zip(direct_grad, implicit_grad)]


def darts_helper(vector, curr, prev, config):
    R = config.darts_alpha
    eps = R / to_vec(vector).norm()

    # positive
    for p, v in zip(curr.trainable_parameters(), vector):
        p.data.add_(eps, v.data)

    losses_p = curr.training_step(curr.cur_batch)
    if not (isinstance(losses_p, tuple) or isinstance(losses_p, list)):
        losses_p = (losses_p,)
        # TODO: Add custom loss aggregation
    losses_p = tuple(loss_p / len(losses_p) for loss_p in losses_p)

    grad_p = None
    for loss_p in losses_p:
        if grad_p is None:
            grad_p = torch.autograd.grad(loss_p, prev.trainable_parameters())
        else:
            grad_p += torch.autograd.grad(loss_p, prev.trainable_parameters())

    # negative
    for p, v in zip(curr.trainable_parameters(), vector):
        p.data.sub_(2 * eps, v.data)

    losses_n = curr.training_step(curr.cur_batch)
    if not (isinstance(losses_n, tuple) or isinstance(losses_n, list)):
        losses_n = (losses_n,)
        # TODO: Add custom loss aggregation
    losses_n = tuple(loss_n / len(losses_n) for loss_n in losses_n)

    grad_n = None
    for loss_n in losses_n:
        if grad_n is None:
            grad_n = torch.autograd.grad(loss_n, prev.trainable_parameters())
        else:
            grad_n += torch.autograd.grad(loss_n, prev.trainable_parameters())

    # reverse weight change
    for p, v in zip(curr.trainable_parameters(), vector):
        p.data.add(eps, v.data)

    implicit_grad = [(x - y).div_(2 * eps) for x, y in zip(grad_n, grad_p)]

    return implicit_grad
