import torch


def grad_from_backward(loss, params, retain_graph=None, create_graph=False):
    # torch.autograd.backward(
    #     loss, retain_graph=retain_graph, create_graph=create_graph, inputs=params
    # )
    # grads = []
    # for param in params:
    #     grads.append(param.grad.clone())
    #     param.grad = None
    # return tuple(grads)
    return torch.autograd.grad(
        loss, params, retain_graph=retain_graph, create_graph=create_graph
    )
