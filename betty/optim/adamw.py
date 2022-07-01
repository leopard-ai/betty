import math

import torch

from betty.optim.optimizer import DifferentiableOptimizerBase


class DifferentiableAdamW(DifferentiableOptimizerBase):
    """
    Differentiable version of PyTorch's
    `AdamW <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#adamw>`_ optimizer.
    All in-place operations are replaced.
    """

    def step(self, params):
        for param_group, param_mapping in zip(self.param_groups, self.param_mappings):

            amsgrad = param_group["amsgrad"]
            beta1, beta2 = param_group["betas"]

            for param_idx in param_mapping:
                p = params[param_idx]

                if p.grad is None:
                    continue
                grad = p.grad

                p = p * (1 - param_group["lr"] * param_group["weight_decay"])
                state = self.state[param_idx]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                state["exp_avg"] = state["exp_avg"] * beta1 + (1 - beta1) * grad
                state["exp_avg_sq"] = (
                    state["exp_avg_sq"] * beta2 + (1 - beta2) * grad * grad
                )

                if amsgrad:
                    state["max_exp_avg_sq"] = torch.max(
                        state["max_exp_avg_sq"], state["exp_avg_sq"]
                    )
                    denom = (
                        state["max_exp_avg_sq"] / math.sqrt(bias_correction2)
                        + param_group["eps"]
                    )
                else:
                    denom = (
                        state["exp_avg_sq"] / math.sqrt(bias_correction2)
                        + param_group["eps"]
                    )

                step_size = param_group["lr"] / bias_correction1
                p.update = step_size * (state["exp_avg"] / denom)

        new_params = tuple(p - p.update for p in params if hasattr(p, "update"))
        for p in params:
            if hasattr(p, "update"):
                del p.update

        return new_params
