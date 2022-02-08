import torch


def maml(loss, params, first_order=False, retain_graph=True):
    grad = torch.autograd.grad(
        loss,
        params,
        create_graph=not first_order,
        retain_graph=retain_graph
    )
    return grad