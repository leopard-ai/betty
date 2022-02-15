import torch


def maml(loss, params, create_graph=True, retain_graph=False, allow_unused=False):
    grad = torch.autograd.grad(
        loss,
        params,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=allow_unused
    )
    return grad
