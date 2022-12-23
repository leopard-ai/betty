# Copyright Sang Keun Choe
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

try:
    import higher

    HAS_HIGHER = True
except ImportError:
    HAS_HIGHER = False

from betty.problems import Problem


# pylint: disable=W0223
class HigherIterativeProblem(Problem):
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
        if not HAS_HIGHER:
            raise ImportError(
                "HigherIterativeProblem requires higher as its dependency. "
                "Run 'pip install higher'. "
                "The higher dependency will be deprecated in the future."
            )
        self.module_state_dict_cache = None
        self.opitmizer_state_dict_cache = None

    def optimizer_step(self, *args, **kwargs):
        self.optimizer.step()

    def functional_one_step_descent(self, batch=None):
        assert not self._fp16 and not self.is_implemented(
            "custom_optimizer_step"
        ), "[!] FP16 training is not supported for IterativeProblem."

        # load data
        if batch is None:
            self.cur_batch = self.get_batch()
            batch = self.cur_batch

        # calculatet loss
        loss, loss_dict = self.get_loss(batch)

        # backward & zero grad & param update
        self.optimizer.step(loss)

        return loss_dict

    def one_step_descent(self, batch=None):
        # load data
        if batch is None:
            self.cur_batch = self.get_batch()
            batch = self.cur_batch

        # calculate loss
        loss, loss_dict = self.get_loss(batch)

        # calculate gradient (a.k.a backward)
        self.backward(
            loss=loss,
            params=self.trainable_parameters(),
            paths=self._paths,
            create_graph=False,
            retain_graph=self._retain_graph,
            allow_unused=self._allow_unused,
        )

        # calculate parameter update
        if self._count % self.gas == 0:
            self.optimizer_step()

            # param callback (e.g., parameter clipping)
            if self.is_implemented("param_callback"):
                self.param_callback(self.trainable_parameters())

            # zero-out grad
            self.zero_grad()

        return loss_dict

    def step_normal(self, global_step=None):
        if self.check_ready():
            # loop start
            if self._inner_loop_start:
                if self.is_implemented("on_inner_loop_start"):
                    self.on_inner_loop_start()
                self._inner_loop_start = False

                if self._roll_back:
                    self.cache_states()

                # patch module & optimizer to support functional
                self.patch_module()
                self.patch_optimizer()

            if self._training:
                self._count += 1

            loss_dict = self.functional_one_step_descent()

            if self.scheduler is not None and not self._roll_back:
                self.scheduler.step()

            if self.log_step > 0 and self._count % self.log_step == 0:
                self.log(loss_dict, global_step)

            if self._training and self._count % (self._unroll_steps * self.gas) == 0:
                for problem in self._parents:
                    idx = problem.children.index(self)
                    problem.ready[idx] = True
                    problem.step_normal(global_step=global_step)

                self._inner_loop_start = True

                self.module_orig.load_state_dict(self.module.state_dict())
                del self.module
                del self.optimizer
                self.module = self.module_orig
                self.optimizer = self.optimizer_orig

            self.ready = [False for _ in range(len(self._children))]

    def patch_module(self):
        """
        Patch PyTorch's native stateful module into the stateless module so as to support
        functional forward that takes params as its input.
        """
        self.module_orig = self.module
        self.module = higher.patch.monkeypatch(
            self.module_orig,
            device=self.device,
            track_higher_grads=not self._first_order,
        )

    def patch_optimizer(self):
        """
        Patch PyTorch's native optimizer by replacing all involved in-place operations to allow
        gradient flow through the parameter update process.
        """
        if self.optimizer is not None:
            self.optimizer_orig = self.optimizer
            self.optimizer = higher.optim.get_diff_optim(
                self.optimizer_orig,
                self.module_orig.parameters(),
                fmodel=self.module,
                device=self.device,
                track_higher_grads=not self._first_order,
            )
            for group in self.optimizer.param_groups:
                group["weight_decay"] = 0
                group["momentum"] = 0

    def cache_states(self):
        self.module_state_dict_cache = self.module.state_dict()
        if self.optimizer is not None:
            self.optimizer_state_dict_cache = self.optimizer.state_dict()

    def recover_states(self):
        self.module.load_state_dict(self.module_state_dict_cache)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(self.optimizer_state_dict_cache)

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
