import torch

from betty.hypergradient.utils import to_vec, sub_with_none


def cg(loss, params, child, config, create_graph=True, retain_graph=False, allow_unused=True):
    # direct grad
    direct_grad = torch.autograd.grad(loss,
                                      params,
                                      create_graph=create_graph,
                                      retain_graph=retain_graph,
                                      allow_unused=allow_unused)

    # implicit grad
    in_loss = child.training_step(child.cur_batch)
    in_grad = torch.autograd.grad(in_loss, child.trainable_parameters(), create_graph=True)

    b = torch.autograd.grad(loss,
                            child.trainable_parameters(),
                            retain_graph=False)
    x = [torch.zeros_like(bb) for bb in b]
    r = [torch.zeros_like(bb).copy_(bb) for bb in b]
    p = [torch.zeros_like(rr).copy_(rr) for rr in r]

    for _ in range(config.cg_iterations):
        hvp = torch.autograd.grad(in_grad, child.parameters(), grad_outputs=p, retain_graph=True)
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

    implicit_grad = torch.autograd.grad(in_grad, params, grad_outputs=x)

    return [sub_with_none(dg, ig) for dg, ig in zip(direct_grad, implicit_grad)]
