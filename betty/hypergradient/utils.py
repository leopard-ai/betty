import math
import torch


def grad(loss, parameters, retain_graph=False, allow_unused=False, is_fsdp=False):
    def get_grad(p):
        return p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)

    if is_fsdp:
        grad_orig = [get_grad(p) for p in parameters]
        torch.autograd.backward(loss, retain_graph=retain_graph, inputs=parameters)

        grads = []
        for p, g in zip(parameters, grad_orig):
            grads.append(get_grad(p) - g)
            p.grad.copy_(g.data)
        return grads
    else:
        return torch.autograd.grad(
            loss, parameters, retain_graph=retain_graph, allow_unused=allow_unused
        )


def get_optimzer_type(optimizer):
    cls_name = type(optimizer).__name__.lower()
    if "adam" in cls_name:
        return "adam"
    elif "rmsprop" in cls_name:
        return "rmsprop"
    return "sgd"


def precondition_sgd(vectors, problem):
    return vectors


def precondition_adam(vectors, problem):
    outputs = []

    params = problem.meta_trainable_parameters()
    for vector, param in zip(vectors, params):
        param_group = problem.get_opt_param_group_for_param(param)
        state = problem.get_opt_state_for_param(param)

        with torch.no_grad():
            beta1, beta2 = param_group["betas"]
            eps = param_group["eps"]
            last_grad = state.get("last_grad", torch.zeros_like(vector))
            exp_avg = state.get("exp_avg", torch.zeros_like(vector))
            exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(vector))
            exp_avg_old = (
                (exp_avg - (1 - beta1) * last_grad) / beta1 if beta1 != 0 else 0
            )
            exp_avg_sq_old = (exp_avg_sq - (1 - beta2) * last_grad * last_grad) / beta2

            scale = (1 - beta1) * beta2 * exp_avg_sq_old - beta1 * (
                1 - beta2
            ) * last_grad * exp_avg_old
            scale /= (torch.sqrt(exp_avg_sq) + eps) ** 3
        out = vector * scale * param_group["lr"]
        outputs.append(out)

    return outputs


def precondition_rmsprop(vectors, problem):
    raise NotImplementedError("SAMA preconditioning for RMSProp is not implemented!")


def precondition_adagrad(vectors, problem):
    raise NotImplementedError("SAMA preconditioning for Adagrad is not implemented!")


def precondition_adadelta(vectors, problem):
    raise NotImplementedError("SAMA preconditioning for Adadelta is not implemented!")


precondition_fn_mapping = {
    "sgd": precondition_sgd,
    "adam": precondition_adam,
    "rmsprop": precondition_rmsprop,
    "adagrad": precondition_adagrad,
    "adadelta": precondition_adadelta,
}


def precondition(vectors, problem):
    optimizer = problem.optimizer
    optimizer_type = get_optimzer_type(optimizer)
    precondition_fn = precondition_fn_mapping[optimizer_type]

    return precondition_fn(vectors, problem)
