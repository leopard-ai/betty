import torch

from betty.hypergradient.utils import to_vec, sub_with_none


def darts(loss, params, path, config, create_graph=True, retain_graph=False, allow_unused=True):
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

    return [sub_with_none(dg, ig) for dg, ig in zip(direct_grad, implicit_grad)]


def darts_helper(vector, curr, prev, config):
    R = config.darts_alpha
    eps = R / to_vec(vector).norm()

    # positie
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

    implicit_grad = [(x - y).div_(2 * eps) for x, y in zip(grad_p, grad_n)]

    return implicit_grad
