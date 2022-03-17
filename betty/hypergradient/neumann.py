import torch

from betty.hypergradient.utils import sub_none

def neumann(loss, params, child, create_graph=True, retain_graph=False, allow_unused=True):
    # direct grad
    direct_grad = torch.autograd.grad(loss,
                                      params,
                                      create_graph=create_graph,
                                      retain_graph=retain_graph,
                                      allow_unused=True)

    # implicit grad
    v1 = torch.autograd.grad(loss,
                             child.trainable_parameters(),
                             retain_graph=True)

    # ! Mabye replace with child.loss by adding self.loss attribute to save computation
    in_loss = child.training_step(child.cur_batch)
    in_grad = torch.autograd.grad(in_loss, child.trainable_parameters(), create_graph=True)
    v2 = approx_inverse_hvp(v1, in_grad, child.trainable_parameters())
    implicit_grad = torch.autograd.grad(in_grad, params, grad_outputs=v2)

    return [sub_none(dg, ig) for dg, ig in zip(direct_grad, implicit_grad)]


def approx_inverse_hvp(v, f, params, iterations=3, alpha=0.01):
    p = v
    for _ in range(iterations):
        hvp = torch.autograd.grad(f, params, grad_outputs=v, retain_graph=True)
        v = [v_i - alpha * hvp_i for v_i, hvp_i in zip(v, hvp)]
        p = [v_i + p_i for v_i, p_i in zip(v, p)]

    return [alpha * p_i for p_i in p]
