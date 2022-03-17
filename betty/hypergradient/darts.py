import torch
from betty.hypergradient.utils import concat

def sub_none(a, b):
    if a is None and b is None:
        raise ValueError
    if a is None:
        return -b
    elif b is None:
        return a
    else:
        return a - b

def darts(loss, params, child, create_graph=True, retain_graph=False, allow_unused=True):
    # direct grad
    direct_grad = torch.autograd.grad(loss,
                                      params,
                                      create_graph=create_graph,
                                      retain_graph=retain_graph,
                                      allow_unused=True)

    # implicit grad
    R = 0.01
    delta = torch.autograd.grad(loss, child.trainable_parameters())
    eps = R / concat(delta).norm()

    # positie
    for p, v in zip(child.trainable_parameters(), delta):
        p.data.add_(eps, v.data)

    losses_p = child.training_step(child.cur_batch)
    if not (isinstance(losses_p, tuple) or isinstance(losses_p, list)):
        losses_p = (losses_p,)
        # TODO: Add custom loss aggregation
    losses_p = tuple(loss_p / len(losses_p) for loss_p in losses_p)

    grad_p = None
    for loss_p in losses_p:
        if grad_p is None:
            grad_p = torch.autograd.grad(loss_p, params)
        else:
            grad_p += torch.autograd.grad(loss_p, params)

    # negative
    for p, v in zip(child.trainable_parameters(), delta):
        p.data.sub_(2 * eps, v.data)

    losses_n = child.training_step(child.cur_batch)
    if not (isinstance(losses_n, tuple) or isinstance(losses_n, list)):
        losses_n = (losses_n,)
        # TODO: Add custom loss aggregation
    losses_n = tuple(loss_n / len(losses_n) for loss_n in losses_n)

    grad_n = None
    for loss_n in losses_n:
        if grad_n is None:
            grad_n = torch.autograd.grad(loss_n, params)
        else:
            grad_n += torch.autograd.grad(loss_n, params)

    # reverse weight change
    for p, v in zip(child.trainable_parameters(), delta):
        p.data.add(eps, v.data)

    implicit_grad = [(x - y).div_(2 * eps) for x, y in zip(grad_p, grad_n)]

    return [sub_none(dg, ig) for dg, ig in zip(direct_grad, implicit_grad)]
