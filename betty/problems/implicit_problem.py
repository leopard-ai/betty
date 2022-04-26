from betty.problems import Problem


#pylint: disable=W0223
class ImplicitProblem(Problem):
    def __init__(self,
                 name,
                 config,
                 module=None,
                 optimizer=None,
                 scheduler=None,
                 train_data_loader=None,
                 device=None):
        super().__init__(name, config, module, optimizer, scheduler, train_data_loader, device)
        self.module_state_dict_cache = None
        self.opitmizer_state_dict_cache = None

    def optimizer_step(self, *args, **kwargs):
        if self.is_implemented('custom_optimizer_step'):
            self.custom_optimizer_step(*args, **kwargs)
        else:
            self.optimizer.step()

    def cache_states(self):
        self.module_state_dict_cache = self.module.state_dict()
        if self.optimizer is not None:
            self.opitmizer_state_dict_cache = self.optimizer.state_dict()

    def recover_states(self):
        self.module.load_state_dict(self.module_state_dict_cache)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(self.opitmizer_state_dict_cache)
        self.opitmizer_state_dict_cache = None

    def parameters(self):
        return list(self.module.parameters())

    def trainable_parameters(self):
        return list(self.module.parameters())
