from abc import abstractmethod

import torch

import betty.utils as utils


class DifferentiableOptimizerBase(torch.optim.Optimizer):
    def __init__(self, optimizer, module):

        self.param_groups = []
        self.state = [None for _ in range(len(list(module.parameters())))]
        self.param_mappings = []

        # initialize param_groups, state, param_mappings from unpatched module and optimizer.
        for init_param_group in optimizer.param_groups:
            param_group = {}
            for key, value in init_param_group.items():
                if key != "params":
                    param_group[key] = value

            param_mapping = []
            for param in init_param_group["params"]:
                param_idx = utils.get_param_index(param, module.parameters())
                param_mapping.append(param_idx)
                self.state[param_idx] = optimizer.state[param]

            self.param_groups.append(param_group)
            self.param_mappings.append(param_mapping)

    @abstractmethod
    def step(self, params):
        raise NotImplementedError
