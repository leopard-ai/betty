import torch


def get_optimzer_type(optimizer):
    cls_name = type(optimizer).__name__.lower()
    if "adam" in cls_name:
        return "adam"
    return "sgd"


def precondition(vectors, problem):
    optimizer = problem.optimizer
    optimizer_type = get_optimzer_type(optimizer)

    if optimizer_type == "sgd":
        return vectors

    outputs = []
    if optimizer_type == "adam":
        params = problem.trainable_parameters()
        for vector, param in zip(vectors, params):
            param_group = problem.get_opt_param_group_for_param(param)
            state = problem.get_opt_state_for_param(param)

            with torch.no_grad():
                beta1, beta2 = param_group["betas"]
                eps = param_group["eps"]
                last_grad = state.get("last_grad", torch.zeros_like(vector))
                exp_avg = state.get("exp_avg", torch.zeros_like(vector))
                exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(vector))
                exp_avg_old = (exp_avg - (1 - beta1) * last_grad) / beta1
                exp_avg_sq_old = (
                    exp_avg_sq - (1 - beta2) * last_grad * last_grad
                ) / beta2

                scale = (1 - beta1) * beta2 * exp_avg_sq_old - beta1 * (
                    1 - beta2
                ) * last_grad * exp_avg_old
                scale /= (torch.sqrt(exp_avg_sq) + eps) ** 3
            out = vector * scale * param_group["lr"]
            outputs.append(out)

    return outputs
