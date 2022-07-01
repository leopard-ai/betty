from betty.optim.optimizer import DifferentiableOptimizerBase


class DifferentiableSGD(DifferentiableOptimizerBase):
    """
    Differentiable version of PyTorch's
    `SGD <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#sgd>`_ optimizer.
    All in-place operations are replaced.
    """

    def step(self, params):
        for param_group, param_mapping in zip(self.param_groups, self.param_mappings):
            weight_decay = param_group["weight_decay"]
            momentum = param_group["momentum"]
            dampening = param_group["dampening"]
            nesterov = param_group["nesterov"]

            for param_idx in param_mapping:
                p = params[param_idx]

                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay != 0:
                    grad = grad + weight_decay * p

                param_state = self.state[param_idx]
                if (
                    "momentum_buffer" not in param_state
                    or param_state["momentum_buffer"] is None
                ):
                    buf = param_state["momentum_buffer"] = grad
                else:
                    buf = param_state["momentum_buffer"]
                    buf = momentum * buf + (1 - dampening) * grad
                    param_state["momentum_buffer"] = buf
                if nesterov:
                    grad = grad + momentum * buf
                else:
                    grad = buf

                p.update = param_group["lr"] * grad

        new_params = tuple(p - p.update for p in params if hasattr(p, "update"))
        for p in params:
            if hasattr(p, "update"):
                del p.update

        return new_params
