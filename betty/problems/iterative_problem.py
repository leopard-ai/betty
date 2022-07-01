# Copyright Sang Keun Choe
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from ast import Import
from copy import deepcopy

import torch

try:
    import functorch

    HAS_FUNCTORCH = True
except ImportError:
    HAS_FUNCTORCH = False

from betty.problems import Problem
import betty.optim as optim


# pylint: disable=W0223
class IterativeProblem(Problem):
    """
    ``IterativeProblem`` is sublassed from ``Problem``.
    """

    def __init__(
        self,
        name,
        config,
        module=None,
        optimizer=None,
        scheduler=None,
        train_data_loader=None,
        device=None,
    ):
        super().__init__(
            name, config, module, optimizer, scheduler, train_data_loader, device
        )
        # functorch installation check
        if not HAS_FUNCTORCH:
            raise ImportError(
                "IterativeProblem requires functorch and PyTorch 1.11. "
                "Run 'pip install functorch'. "
                "The functorch dependency will be deprecated in the future."
            )
        # functional modules
        self.fmodule = None
        self.params = None
        self.buffers = None

        self.params_cache = None
        self.buffers_cache = None
        self.opitmizer_state_dict_cache = None

    def initialize(self, engine_config):
        super().initialize(engine_config=engine_config)
        # patch module to be functional so that gradient flows through param update
        # optimizer & scheduler should accordingly be patched as module gets patched
        self.initialize_optimizer_state()
        self.patch_modules()
        self.patch_optimizer()
        self.patch_scheduler()

    def optimizer_step(self, *args, **kwargs):
        assert (
            not self._fp16
        ), "[!] FP16 training is not supported for IterativeProblem."
        if self.is_implemented("custom_optimizer_step"):
            self.params = self.custom_optimizer_step(*args, **kwargs)
        else:
            self.params = self.optimizer.step(self.params)

    def initialize_optimizer_state(self):
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                param.grad = torch.zeros_like(param.data)
        self.optimizer.step()

    def patch_modules(self):
        """
        Patch PyTorch's native stateful module into the stateless module so as to support
        functional forward that takes params as its input.
        """
        fmodule, params, buffers = functorch.make_functional_with_buffers(self.module)
        self.fmodule = fmodule
        self.params = params
        self.buffers = buffers

    def patch_optimizer(self):
        """
        Patch PyTorch's native optimizer by replacing all involved in-place operations to allow
        gradient flow through the parameter update process.
        """
        if self.optimizer is not None:
            self.optimizer = optim.patch_optimizer(self.optimizer, self.module)

    def patch_scheduler(self):
        """
        Patch scheduler to be compatible with the patched optimizer
        """
        if self.scheduler is not None:
            self.scheduler = optim.patch_scheduler(self.scheduler, self.optimizer)

    def cache_states(self):
        # TODO: replace deepcopy with state_dict
        self.params_cache = deepcopy(self.params)
        self.buffers_cache = deepcopy(self.buffers)
        if self.optimizer is not None:
            self.opitmizer_state_dict_cache = deepcopy(self.optimizer.state)

    def recover_states(self):
        # TODO: change loading mechanism based on state_dict
        self.params, self.params_cache = self.params_cache, None
        self.buffers, self.buffers_cache = self.buffers_cache, None
        if self.optimizer is not None:
            self.optimizer.state = self.opitmizer_state_dict_cache
        self.opitmizer_state_dict_cache = None

    def parameters(self):
        return self.params

    def trainable_parameters(self):
        return self.params
