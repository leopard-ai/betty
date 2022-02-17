import math
from tkinter import W

import torch

def fadam(
    params,
    param_mapping,
    param_groups,
    states
):
    for group_idx, group_mapping in enumerate(param_mapping):
        group = param_groups[group_idx]

        amsgrad = group['amsgrad']
        beta1, beta2 = group['betas']
        weight_decay = group['weight_decay']

        for param_idx in group_mapping:
            p = params[param_idx]

            if p.gradient is None:
                continue
            grad = p.gradient

            state = states[param_idx]

            state['step'] += 1
            bias_correction1 = 1 - beta1**state['step']
            bias_correction2 = 1 - beta2**state['step']

            if weight_decay != 0:
                grad = grad + (weight_decay * p)

            state['exp_avg'] = state['exp_avg'] * beta1 + (1 - beta1) * grad
            state['exp_avg_sq'] =  state['exp_avg_sq'] * beta2 + (1 - beta2) * grad * grad

            if amsgrad:
                state['max_exp_avg_sq'] = torch.max(state['max_exp_avg_sq'], state['exp_avg_sq'])
                denom = state['max_exp_avg_sq'].sqrt() / math.sqrt(bias_correction2) + group['eps']
            else:
                denom = state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2) + group['eps']

            step_size = group['lr'] / bias_correction1
            p.update = step_size * (state['exp_avg'] / denom)

    out = tuple(p - p.update for p in params if hasattr(p, 'update'))
    for p in params:
        if hasattr(p, 'update'):
            del p.update
    return out
