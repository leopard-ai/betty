from copy import deepcopy

import torch
import functorch

from betty.problems import Problem
import betty.optim as optim


#pylint: disable=W0223
class IterativeProblem(Problem):
    def __init__(self,
                 name,
                 config,
                 module=None,
                 optimizer=None,
                 scheduler=None,
                 train_data_loader=None,
                 device=None):
        super().__init__(name, config, module, optimizer, scheduler, train_data_loader, device)
        # functional modules
        self.fmodule = None
        self.params = None
        self.buffers = None

        self.params_cache = None
        self.buffers_cache = None
        self.opitmizer_state_dict_cache = None
        self.scheduler_state_dict_cache = None

    def initialize(self):
        super().initialize()
        # patch module to be functional so that gradient flows through param update
        # optimizer & scheduler should accordingly be patched as module gets patched
        self.initialize_optimizer_state()
        self.patch_models()
        self.patch_optimizer()
        self.patch_scheduler()

    def optimizer_step(self, *args, **kwargs):
        if self.is_implemented('custom_optimizer_step'):
            params = self.custom_optimizer_step(*args, **kwargs)
        else:
            params = self.optimizer.step(self.params)

        self.params = params

    def initialize_optimizer_state(self):
        """[summary]
        Initialize optimizer state
        """
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                param.grad = torch.zeros_like(param.data)
        self.optimizer.step()

    def patch_models(self):
        """[summary]
        Patch models to support functional forward that takes params as an input
        """
        fmodule, params, buffers = functorch.make_functional_with_buffers(self.module)
        self.fmodule = fmodule
        self.params = params
        self.buffers = buffers

    def patch_optimizer(self):
        """[summary]
        Patch optimizer to avoid in-place operations so that gradient flows through param update.
        Raises:
            NotImplementedError: [description]
        """
        if self.optimizer is not None:
            self.optimizer = optim.patch_optimizer(self.optimizer, self.module)

    def patch_scheduler(self):
        """[summary]
        Patch scheduler to work on patched optimizer
        """
        if self.scheduler is not None:
            self.scheduler = optim.patch_scheduler(self.scheduler, self.optimizer)

    def cache_states(self):
        # TODO: replace deepcopy with state_dict
        self.params_cache = deepcopy(self.params)
        self.buffers_cache = deepcopy(self.buffers)
        if self.optimizer is not None:
            self.opitmizer_state_dict_cache = deepcopy(self.optimizer.state)
        if self.scheduler is not None:
            self.scheduler_state_dict_cache = self.scheduler.state_dict()

    def recover_states(self):
        # TODO: change loading mechanism based on state_dict
        self.params, self.params_cache = self.params_cache, None
        self.buffers, self.buffers_cache = self.buffers_cache, None
        if self.optimizer is not None:
            self.optimizer.state = self.opitmizer_state_dict_cache
        if self.scheduler is not None:
            self.scheduler.load_state_dict(self.scheduler_state_dict_cache)
        self.opitmizer_state_dict_cache = None
        self.scheduler_state_dict_cache = None

    def parameters(self):
        return self.params

    def trainable_parameters(self):
        return self.params
