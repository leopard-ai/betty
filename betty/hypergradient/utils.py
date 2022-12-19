import math
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

            step = state.get("step", 0)
            exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(vector))
            beta2 = param_group["betas"][1]
            eps = param_group["eps"]
            denom = torch.add(
                torch.sqrt(torch.mean(beta2 * exp_avg_sq)), eps
            ) / math.sqrt(1 - beta2**step)

            outputs.append(vector / denom)

    return outputs
