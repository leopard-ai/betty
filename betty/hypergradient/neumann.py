import torch

from betty.hypergradient.utils import sub_with_none


def neumann(loss, params, path, config, create_graph=True, retain_graph=False, allow_unused=True):
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
        implicit_grad = neumann_helper(implicit_grad, path[i], path[i+1], config)

    return [sub_with_none(dg, ig) for dg, ig in zip(direct_grad, implicit_grad)]


def neumann_helper(vector, curr, prev, config):
    # ! Mabye replace with child.loss by adding self.loss attribute to save computation
    in_loss = curr.training_step(curr.cur_batch)
    in_grad = torch.autograd.grad(in_loss, curr.trainable_parameters(), create_graph=True)
    v2 = approx_inverse_hvp(vector, in_grad, curr.trainable_parameters(),
                            iterations=config.neumann_iterations,
                            alpha=config.neumann_alpha)
    implicit_grad = torch.autograd.grad(in_grad, prev.trainable_parameters(), grad_outputs=v2)

    return implicit_grad


def approx_inverse_hvp(v, f, params, iterations=3, alpha=1.):
    p = v
    for _ in range(iterations):
        hvp = torch.autograd.grad(f, params, grad_outputs=v, retain_graph=True)
        v = [v_i - alpha * hvp_i for v_i, hvp_i in zip(v, hvp)]
        p = [v_i + p_i for v_i, p_i in zip(v, p)]

    return [alpha * p_i for p_i in p]
