import torch
from betty.hypergradient.utils import concat

def darts(loss, params, child, create_graph=True, retain_graph=False, allow_unused=False):
    # direct grad
    direct_grad = torch.autograd.grad(loss,
                                      params,
                                      create_graph=create_graph,
                                      retain_graph=retain_graph,
                                      allow_unused=allow_unused)

    # implicit grad
    delta = 0.01 / concat(child.grad_temp).norm()

    # positie
    for p, v in zip(child.params, child.grad_temp):
        p.data.add_(delta, v.data)

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
    for p, v in zip(child.params, child.grad_temp):
        p.data.sub_(2 * delta, v.data)

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
    for p, v in zip(child.params, child.grad_temp):
        p.data.add(delta, v.data)

    implicit_grad = [(x - y).div_(2 * delta) for x, y in zip(grad_p, grad_n)]

    return direct_grad - implicit_grad
    