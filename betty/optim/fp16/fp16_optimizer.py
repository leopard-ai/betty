import math

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from betty.utils import get_grad_norm


class FP16_Optimizer:
    def __init__(self,
                 init_optimizer,
                 dynamic_loss_scale=False,
                 static_loss_scale=1.0,
                 initial_dynamic_scale=2**32,
                 clip_grad=0.):

        if not torch.cuda.is_available():
            raise SystemError("Cannot use fp16 without CUDA.")
        self.optimizer = init_optimizer

        # param flattened by groups
        self.fp16_groups = []
        self.fp16_groups_flat = []
        self.fp32_groups_flat = []

        for i, param_group in enumerate(self.optimizer.param_groups):
            # push this group to list before modify
            self.fp16_groups.append(param_group['params'])
            # init fp16 weight buffer, flattened
            self.fp16_groups_flat.append(
                _flatten_dense_tensors([p.clone().detach() for p in self.fp16_groups[i]])
            )
            # set model fp16 weight to slices of flattened buffer
            updated_params = _unflatten_dense_tensors(self.fp16_groups_flat[i], self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data
            # init master weight
            self.fp32_groups_flat.append(
                self.fp16_groups_flat[i].clone().float().detach()
            )
            # modify optimizer to have flat master weight
            self.fp32_groups_flat[i].requires_grad = True
            param_group['params'] = [self.fp32_groups_flat[i]]

        if dynamic_loss_scale:
            self.dynamic_loss_scale = True
            self.cur_iter = 0
            self.last_overflow_iter = -1
            self.scale_factor = 2

            self.cur_scale = initial_dynamic_scale
            self.scale_window = 1000
            self.min_loss_scale = 1
        else:
            self.dynamic_loss_scale = False
            self.cur_iter = 0
            self.cur_scale = static_loss_scale

        self.clip_grad = clip_grad
        self.clip_grad_norm = torch.nn.utils.clip_grad_norm_
        self.norm_type = 2

        self.overflow = False
        self.initialize_optimizer_states()

    def initialize_optimizer_states(self):
        for i, _ in enumerate(self.fp16_groups):
            self.fp32_groups_flat[i].grad = torch.zeros(
                self.fp32_groups_flat[i].size(),
                device=self.fp32_groups_flat[i].device
            )

        self.optimizer.step()

        for i, _ in enumerate(self.fp16_groups):
            self.fp32_groups_flat[i].grad = None

        return

    def zero_grad(self, set_grads_to_none=True):
        for group in self.fp16_groups:
            for p in group:
                if set_grads_to_none:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

    def step(self, closure=None):
        """
        Not supporting closure.
        """

        fp16_params = []
        for i, fp16_group in enumerate(self.fp16_groups):
            fp16_params.extend([p for p in fp16_group if p.grad is not None])
        self.overflow = self.has_overflow(fp16_params)
        prev_scale = self.cur_scale
        self._update_scale(self.overflow)

        if self.overflow:
            print(f"Overflow detected. Skipping step. Attempted loss scale: {prev_scale}, reducing to {self.cur_scale}")
            for i, fp16_group in enumerate(self.fp16_groups):
                for p in fp16_group:
                    p.grad = None
            return self.overflow

        grads_groups_flat = []
        for i, group in enumerate(self.fp16_groups):
            data_type = self.fp32_groups_flat[i].dtype

            grads_groups_flat.append(
                _flatten_dense_tensors([
                    torch.zeros(p.size(),
                                dtype=data_type,
                                device=p.device)
                    if p.grad is None else p.grad.to(data_type) for p in group
                ])
            )

            for p in group:
                p.grad = None

            self.fp32_groups_flat[i].grad = grads_groups_flat[i]

        all_groups_norm = get_grad_norm(self.fp32_groups_flat)
        self.unscale_and_clip_grads(grads_groups_flat, [all_groups_norm])

        self.optimizer.step()

        for group in self.fp32_groups_flat:
            group.grad = None

        for i, fp16_group in enumerate(self.fp16_groups):
            updated_params = _unflatten_dense_tensors(self.fp32_groups_flat[i],
                                                      fp16_group)
            for p, q in zip(fp16_group, updated_params):
                p.data.copy_(q.data)

        return self.overflow

    def unscale_and_clip_grads(self, grad_groups_flat, norm_groups, apply_scale=True):
        total_norm = 0.0
        for norm in norm_groups:
            total_norm += norm**2.0
        total_norm = math.sqrt(total_norm)

        # compute combined scale factor for this group
        combined_scale = self.cur_scale
        if self.clip_grad > 0.:
            # norm is in fact norm*scale
            clip = ((total_norm / self.cur_scale) + 1e-6) / self.clip_grad
            if clip > 1:
                combined_scale = clip * self.cur_scale

        if apply_scale:
            for grad in grad_groups_flat:
                grad.data.mul_(1. / combined_scale)

        return combined_scale

    def _update_scale(self, skip):
        if self.dynamic_loss_scale:
            if skip:
                self.cur_scale = max(self.cur_scale / self.scale_factor,
                                     self.min_loss_scale)
                self.last_overflow_iter = self.cur_iter
            else:
                # Ensure self.scale_window updates since last overflow
                stable_interval = (self.cur_iter - self.last_overflow_iter) - 1
                if (stable_interval > 0) and (stable_interval % self.scale_window == 0):
                    self.cur_scale *= self.scale_factor
        else:
            if skip:
                print("Grad overflow on iteration: %s", self.cur_iter)
                print("Using static loss scale of: %s", self.cur_scale)
        self.cur_iter += 1
        return

    def has_overflow(self, params=None):
        for param in params:
            if param.grad is not None and self._has_inf_or_nan(param.grad.data):
                return True
        return False

    @staticmethod
    def _has_inf_or_nan(x):
        try:
            cpu_sum = float(x.float().sum())
        except RuntimeError as instance:
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
                return True
            return False

    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.

        .. code:: python

            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['cur_scale'] = self.cur_scale
        state_dict['cur_iter'] = self.cur_iter
        if state_dict['dynamic_loss_scale']:
            state_dict['last_overflow_iter'] = self.last_overflow_iter
            state_dict['scale_factor'] = self.scale_factor
            state_dict['scale_window'] = self.scale_window
        state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        state_dict['fp32_groups_flat'] = self.fp32_groups_flat
        state_dict['clip_grad'] = self.clip_grad

        return state_dict

    def load_state_dict(self, state_dict, load_optimizer_states=True):
        """
        Loads a state_dict created by an earlier call to state_dict().
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.

        .. code:: python

            model = torch.nn.Linear(D_in, D_out).cuda().half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
        self.cur_scale = state_dict['cur_scale']
        self.cur_iter = state_dict['cur_iter']
        if state_dict['dynamic_loss_scale']:
            self.last_overflow_iter = state_dict['last_overflow_iter']
            self.scale_factor = state_dict['scale_factor']
            self.scale_window = state_dict['scale_window']
        if load_optimizer_states:
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.clip_grad = state_dict['clip_grad']

    def __repr__(self):
        return repr(self.optimizer)
