import torch

from betty.hypergradient.utils import sub_with_none


def reinforce(loss, params, path, config, create_graph=True, retain_graph=False, allow_unused=True):
    # direct grad
    direct_grad = torch.autograd.grad(loss,
                                      params,
                                      create_graph=create_graph,
                                      retain_graph=retain_graph,
                                      allow_unused=allow_unused)

    # implicit grad
    implicit_grad = torch.autograd.grad(loss,
                                        path[1].trainable_parameters(),
                                        retain_graph=False)
    # TODO: recursion
    #for i in range(1, len(path)-1):
    #    implicit_grad = darts_helper(implicit_grad, path[i], path[i+1], config)

    return [sub_with_none(dg, ig) for dg, ig in zip(direct_grad, implicit_grad)]


def reinforce_helper(vector, curr, prev, config):
    raise NotImplementedError