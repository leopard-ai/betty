import torch

from betty.hypergradient.utils import neg_with_none, to_vec, add_with_none


def cg(loss, params, path, config, create_graph=True, retain_graph=False, allow_unused=True):
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
    for i in range(1, len(path)-1):
        implicit_grad = cg_helper(implicit_grad, path[i], path[i+1], config)

    return [add_with_none(dg, ig) for dg, ig in zip(direct_grad, implicit_grad)]


def cg_helper(vector, curr, prev, config):
    in_loss = curr.training_step(curr.cur_batch)
    in_grad = torch.autograd.grad(in_loss, curr.trainable_parameters(), create_graph=True)

    x = [torch.zeros_like(vi) for vi in vector]
    r = [torch.zeros_like(vi).copy_(vi) for vi in vector]
    p = [torch.zeros_like(rr).copy_(rr) for rr in r]

    for _ in range(config.cg_iterations):
        hvp = torch.autograd.grad(in_grad, curr.parameters(), grad_outputs=p, retain_graph=True)
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

    implicit_grad = torch.autograd.grad(in_grad, prev.trainable_parameters(), grad_outputs=x)
    implicit_grad = [neg_with_none(ig) for ig in implicit_grad]

    return implicit_grad
