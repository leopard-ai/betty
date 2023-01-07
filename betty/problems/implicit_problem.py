# Copyright Sang Keun Choe
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from betty.problems import Problem


# pylint: disable=W0223
class ImplicitProblem(Problem):
    """
    ``ImplicitProblem`` is sublassed from ``Problem``.
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
        self.module_state_dict_cache = None
        self.opitmizer_state_dict_cache = None

    def optimizer_step(self, *args, **kwargs):
        if self.is_implemented("custom_optimizer_step"):
            assert (
                not self._is_default_fp16()
            ), "[!] FP16 training is not supported for custom optimizer step."
            if self.gradient_clipping > 0.0:
                self.clip_grad()
            self.custom_optimizer_step(*args, **kwargs)
        else:
            if self._is_default_fp16():
                if self.gradient_clipping > 0.0:
                    self.scaler.unscale_(self.optimizer)
                    self.clip_grad()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.gradient_clipping > 0.0:
                    self.clip_grad()
                self.optimizer.step()

    def cache_states(self):
        self.module_state_dict_cache = self.module.state_dict()
        if self.optimizer is not None:
            self.opitmizer_state_dict_cache = self.optimizer.state_dict()

    def recover_states(self):
        self.module.load_state_dict(self.module_state_dict_cache)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(self.opitmizer_state_dict_cache)
        self.module_state_dict_cache = None
        self.opitmizer_state_dict_cache = None

    def parameters(self):
        return list(self.module.parameters())

    def trainable_parameters(self):
        return list(self.module.parameters())

    def train(self):
        super().train()
        self.module.train()

    def eval(self):
        super().eval()
        self.module.eval()
