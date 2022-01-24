import torch
from torch.autograd import grad as torch_grad
from torch import Tensor
from hypergrad import CG_torch
from typing import List, Callable


# noinspection PyUnusedLocal
def reverse_unroll(params: List[Tensor],
                   hparams: List[Tensor],
                   outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
                   set_grad=True) -> List[Tensor]:
    """
    Computes the hypergradient by backpropagating through a previously employed inner solver procedure.

    Args:
        params: the output of a torch differentiable inner solver (it must depend on hparams in the torch graph)
        hparams: the outer variables (or hyperparameters), each element needs requires_grad=True
        outer_loss: computes the outer objective taking parameters and hyperparameters as inputs
        set_grad: if True set t.grad to the hypergradient for every t in hparams

    Returns:
        the list of hypergradients for each element in hparams
    """
    o_loss = outer_loss(params, hparams)
    grads = torch.autograd.grad(o_loss, hparams, retain_graph=True)
    if set_grad:
        update_tensor_grads(hparams, grads)
    return grads


# noinspection PyUnusedLocal
def reverse(params_history: List[List[Tensor]],
            hparams: List[Tensor],
            update_map_history: List[Callable[[List[Tensor], List[Tensor]], List[Tensor]]],
            outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
            set_grad=True) -> List[Tensor]:
    """
    Computes the hypergradient by recomputing and backpropagating through each inner update
    using the inner iterates and the update maps previously employed by the inner solver.
    Similarly to checkpointing, this allows to save memory w.r.t. reverse_unroll by increasing computation time.
    Truncated reverse can be performed by passing only part of the trajectory information, i.e. only the
    last k inner iterates and updates.

    Args:
        params_history: the inner iterates (from first to last)
        hparams: the outer variables (or hyperparameters), each element needs requires_grad=True
        update_map_history: updates used to solve the inner problem (from first to last)
        outer_loss: computes the outer objective taking parameters and hyperparameters as inputs
        set_grad: if True set t.grad to the hypergradient for every t in hparams

    Returns:
         the list of hypergradients for each element in hparams

    """
    params_history = [[w.detach().requires_grad_(True) for w in params] for params in params_history]
    o_loss = outer_loss(params_history[-1], hparams)
    grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params_history[-1], hparams)

    alphas = grad_outer_w
    grads = [torch.zeros_like(w) for w in hparams]
    K = len(params_history) - 1
    for k in range(-2, -(K + 2), -1):
        w_mapped = update_map_history[k + 1](params_history[k], hparams)
        bs = grad_unused_zero(w_mapped, hparams, grad_outputs=alphas, retain_graph=True)
        grads = [g + b for g, b in zip(grads, bs)]
        alphas = torch_grad(w_mapped, params_history[k], grad_outputs=alphas)

    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]
    if set_grad:
        update_tensor_grads(hparams, grads)

    return grads


def fixed_point(params: List[Tensor],
                hparams: List[Tensor],
                K: int ,
                fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
                outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
                tol=1e-10,
                set_grad=True,
                stochastic=False) -> List[Tensor]:
    """
    Computes the hypergradient by applying K steps of the fixed point method (it can end earlier when tol is reached).

    Args:
        params: the output of the inner solver procedure.
        hparams: the outer variables (or hyperparameters), each element needs requires_grad=True
        K: the maximum number of fixed point iterations
        fp_map: the fixed point map which defines the inner problem
        outer_loss: computes the outer objective taking parameters and hyperparameters as inputs
        tol: end the method earlier when  the normed difference between two iterates is less than tol
        set_grad: if True set t.grad to the hypergradient for every t in hparams
        stochastic: set this to True when fp_map is not a deterministic function of its inputs

    Returns:
        the list of hypergradients for each element in hparams
    """

    params = [w.detach().requires_grad_(True) for w in params]
    o_loss = outer_loss(params, hparams)
    grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)

    if not stochastic:
        w_mapped = fp_map(params, hparams)

    vs = [torch.zeros_like(w) for w in params]
    vs_vec = cat_list_to_tensor(vs)
    for k in range(K):
        vs_prev_vec = vs_vec

        if stochastic:
            w_mapped = fp_map(params, hparams)
            vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=False)
        else:
            vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=True)

        vs = [v + gow for v, gow in zip(vs, grad_outer_w)]
        vs_vec = cat_list_to_tensor(vs)
        if float(torch.norm(vs_vec - vs_prev_vec)) < tol:
            break

    if stochastic:
        w_mapped = fp_map(params, hparams)

    grads = torch_grad(w_mapped, hparams, grad_outputs=vs, allow_unused=True)
    grads = [g + v if g is not None else v for g, v in zip(grads, grad_outer_hparams)]

    if set_grad:
        update_tensor_grads(hparams, grads)

    return grads


def CG(params: List[Tensor],
       hparams: List[Tensor],
       K: int ,
       fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
       outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
       tol=1e-10,
       set_grad=True,
       stochastic=False) -> List[Tensor]:
    """
     Computes the hypergradient by applying K steps of the conjugate gradient method (CG).
     It can end earlier when tol is reached.

     Args:
         params: the output of the inner solver procedure.
         hparams: the outer variables (or hyperparameters), each element needs requires_grad=True
         K: the maximum number of conjugate gradient iterations
         fp_map: the fixed point map which defines the inner problem
         outer_loss: computes the outer objective taking parameters and hyperparameters as inputs
         tol: end the method earlier when the norm of the residual is less than tol
         set_grad: if True set t.grad to the hypergradient for every t in hparams
         stochastic: set this to True when fp_map is not a deterministic function of its inputs

     Returns:
         the list of hypergradients for each element in hparams
     """
    params = [w.detach().requires_grad_(True) for w in params]
    o_loss = outer_loss(params, hparams)
    grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)

    if not stochastic:
        w_mapped = fp_map(params, hparams)

    def dfp_map_dw(xs):
        if stochastic:
            w_mapped_in = fp_map(params, hparams)
            Jfp_mapTv = torch_grad(w_mapped_in, params, grad_outputs=xs, retain_graph=False)
        else:
            Jfp_mapTv = torch_grad(w_mapped, params, grad_outputs=xs, retain_graph=True)
        return [v - j for v, j in zip(xs, Jfp_mapTv)]

    vs = CG_torch.cg(dfp_map_dw, grad_outer_w, max_iter=K, epsilon=tol)  # K steps of conjugate gradient

    if stochastic:
        w_mapped = fp_map(params, hparams)

    grads = torch_grad(w_mapped, hparams, grad_outputs=vs)
    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]

    if set_grad:
        update_tensor_grads(hparams, grads)

    return grads


def CG_normaleq(params: List[Tensor],
                hparams: List[Tensor],
                K: int ,
                fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
                outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
                tol=1e-10,
                set_grad=True) -> List[Tensor]:
    """ Similar to CG but the conjugate gradient is applied on the normal equation (has a higher time complexity)"""
    params = [w.detach().requires_grad_(True) for w in params]
    o_loss = outer_loss(params, hparams)
    grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)

    w_mapped = fp_map(params, hparams)

    def dfp_map_dw(xs):
        Jfp_mapTv = torch_grad(w_mapped, params, grad_outputs=xs, retain_graph=True)
        v_minus_Jfp_mapTv = [v - j for v, j in zip(xs, Jfp_mapTv)]

        # normal equation part
        Jfp_mapv_minus_Jfp_mapJfp_mapTv = jvp(lambda _params: fp_map(_params, hparams), params, v_minus_Jfp_mapTv)
        return [v - vv for v, vv in zip(v_minus_Jfp_mapTv, Jfp_mapv_minus_Jfp_mapJfp_mapTv)]

    v_minus_Jfp_mapv = [g - jfp_mapv for g, jfp_mapv in zip(grad_outer_w, jvp(
        lambda _params: fp_map(_params, hparams), params, grad_outer_w))]
    vs = CG_torch.cg(dfp_map_dw, v_minus_Jfp_mapv, max_iter=K, epsilon=tol)  # K steps of conjugate gradient

    grads = torch_grad(w_mapped, hparams, grad_outputs=vs, allow_unused=True)
    grads = [g + v if g is not None else v for g, v in zip(grads, grad_outer_hparams)]

    if set_grad:
        update_tensor_grads(hparams, grads)

    return grads


def neumann(params: List[Tensor],
            hparams: List[Tensor],
            K: int ,
            fp_map: Callable[[List[Tensor], List[Tensor]], List[Tensor]],
            outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
            tol=1e-10,
            set_grad=True) -> List[Tensor]:
    """ Saves one iteration from the fixed point method"""

    # from https://arxiv.org/pdf/1803.06396.pdf,  should return the same gradient of fixed point K+1
    params = [w.detach().requires_grad_(True) for w in params]
    o_loss = outer_loss(params, hparams)
    grad_outer_w, grad_outer_hparams = get_outer_gradients(o_loss, params, hparams)

    w_mapped = fp_map(params, hparams)
    vs, gs = grad_outer_w, grad_outer_w
    gs_vec = cat_list_to_tensor(gs)
    for k in range(K):
        gs_prev_vec = gs_vec
        vs = torch_grad(w_mapped, params, grad_outputs=vs, retain_graph=True)
        gs = [g + v for g, v in zip(gs, vs)]
        gs_vec = cat_list_to_tensor(gs)
        if float(torch.norm(gs_vec - gs_prev_vec)) < tol:
            break

    grads = torch_grad(w_mapped, hparams, grad_outputs=gs)
    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]
    if set_grad:
        update_tensor_grads(hparams, grads)
    return grads


def exact(opt_params_f: Callable[[List[Tensor]], List[Tensor]],
          hparams: List[Tensor],
          outer_loss: Callable[[List[Tensor], List[Tensor]], Tensor],
          set_grad=True) -> List[Tensor]:
    """
    Computes the exact hypergradient using backpropagation and exploting the closed form torch differentiable function
    that computes the optimal parameters given the hyperparameters (opt_params_f).
    """
    grads = torch_grad(outer_loss(opt_params_f(hparams), hparams), hparams)
    if set_grad:
        update_tensor_grads(hparams, grads)
    return grads


# UTILS

def grd(a, b):
    return torch.autograd.grad(a, b, create_graph=True, retain_graph=True)


def list_dot(l1, l2):  # extended dot product for lists
    return torch.stack([(a*b).sum() for a, b in zip(l1, l2)]).sum()


def jvp(fp_map, params, vs):
    dummy = [torch.ones_like(phw).requires_grad_(True) for phw in fp_map(params)]
    g1 = grd(list_dot(fp_map(params), dummy), params)
    return grd(list_dot(vs, g1), dummy)


def get_outer_gradients(outer_loss, params, hparams, retain_graph=True):
    grad_outer_w = grad_unused_zero(outer_loss, params, retain_graph=retain_graph)
    grad_outer_hparams = grad_unused_zero(outer_loss, hparams, retain_graph=retain_graph)

    return grad_outer_w, grad_outer_hparams


def cat_list_to_tensor(list_tx):
    return torch.cat([xx.view([-1]) for xx in list_tx])


def update_tensor_grads(hparams, grads):
    for l, g in zip(hparams, grads):
        if l.grad is None:
            l.grad = torch.zeros_like(l)
        if g is not None:
            l.grad += g


def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs, allow_unused=True,
                                retain_graph=retain_graph, create_graph=create_graph)

    def grad_or_zeros(grad, var):
        return torch.zeros_like(var) if grad is None else grad

    return tuple(grad_or_zeros(g, v) for g, v in zip(grads, inputs))


