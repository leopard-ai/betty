# Copyright Sang Keun Choe
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Union, Optional, Callable, Dict
from torch.optim.optimizer import Optimizer
from torch.optim import SGD
from itertools import chain
import torch
from torch import nn, Tensor
from typing import List, Union, Any, Iterable
from collections import OrderedDict
from torch.nn.modules.batchnorm import _BatchNorm

from betty.problems import Problem


class MetaModule(nn.Module):
    """
    This is a subclass of nn.Module designed for Meta-Learning.
    The key point of this class is that we will do some operations on nn.Parameter directly with gradient,
    but after doing some operation like `param - lr * grad`, the type of `param` will change from `nn.Parameter`
    to 'torch.Tensor'. If we re-cast the tensor to nn.Parameter, the grad will disappear.
    Currently, the only solution in pytroch frame is to cast the type of all parameters from `nn.Parameter` to `autograd.Variable`
    with `require_grad=True` and then call `register_buffer` function to regist the tensor so that we can use it as normal.
    Expample:
    >>> class Net(nn.Module):
    >>>     ...
    >>> class MetaNet(Net,MetaModule):
    >>>     pass # just extends two class to create MetaNet
    """

    _ignore_buffers = {}
    _ignore_modules = {_BatchNorm}

    def set_params(self, params: Iterable[torch.Tensor]):
        """
        set params' value with order
        """
        for piter, param in zip(
            MetaModule._name_params(self, with_module=True), params
        ):  # type: Any,torch.Tensor
            name, val, mmodule = piter  # type: str,nn.Parameter,nn.Module
            if param is None:
                continue
            if not isinstance(param, torch.autograd.Variable):
                param = torch.autograd.Variable(param, requires_grad=True)
            val.detach()
            # mmodule.register_buffer(name, param)
            setattr(mmodule, name, param)

    def update_params(self, lr: float, grads: Iterable[torch.Tensor]):
        """
        `param - lr*grad` param by param
        """
        nparams = []
        for param, grad in zip(self.params(), grads):
            if grad is None:
                nparams.append(None)
            else:
                nparams.append(param - lr * grad)
        self.set_params(nparams)

    def name_params(self, with_module=False):
        for val in MetaModule._name_params(self, with_module):
            yield val

    def params(self):
        for _, val in self.name_params(with_module=False):
            yield val

    def __setattr__(self, name: str, value: Union[Tensor, "Module"]) -> None:
        super().__setattr__(name, value)

        if isinstance(value, nn.Module):
            MetaModule._move_params_to_buffer(self)

    @staticmethod
    def _move_params_to_buffer(value: nn.Module):
        """
        cast all params' type from `Paramter` to `Variable`
        """
        od = OrderedDict()
        for k, v in value._parameters.items():  # type:str,nn.Parameter
            if v is None:
                od[k] = None
            else:
                od[k] = torch.autograd.Variable(v.data, requires_grad=True)
        for k in od:
            value._parameters.pop(k)
        for k, v in od.items():
            value.register_buffer(k, v)

        for v in value.children():
            MetaModule._move_params_to_buffer(v)

    @staticmethod
    def _name_params(module: nn.Module, with_module=False, prefix=""):
        """yield all params with raw name(without module prefix)"""
        memo = set()
        for mname, mmodule in module.named_children():
            if mmodule == module:
                continue
            for name, val, mmmodule in MetaModule._name_params(
                mmodule, with_module=True, prefix=prefix
            ):
                whole_name = ".".join([prefix, mname, name]).lstrip(".")
                memo.add(whole_name)

                # the name is reletive to module, cause when set or update params, setattr(module, name, val) is called.
                if with_module:
                    yield name, val, mmmodule
                else:
                    yield name, val

        # In MetaModule, there will be no Paramter, all Paramters will be cast to `autograd.Variable` and be registed in
        # buffers, so we only yield `named_buffers` without `named_parameters`.

        for name, val in chain(module.named_buffers(recurse=False)):
            if name in memo:
                continue

            if any([isinstance(module, cls) for cls in MetaModule._ignore_modules]):
                continue

            ignore_names = {}
            for cls in MetaModule._ignore_buffers:
                if isinstance(module, cls):
                    ignore_names = MetaModule._ignore_buffers[cls]
                    break

            pure_name = name.split(".")[-1]
            if pure_name in ignore_names:
                continue

            if with_module:
                yield pure_name, val, module
            else:
                yield pure_name, val

    @staticmethod
    def _params(module: nn.Module):
        """yield all params with raw name(without module prefix)"""
        for _, val in MetaModule._name_params(module):
            yield val

    def zero_grad(self) -> None:
        super(MetaModule, self).zero_grad()
        for name, p in self.named_buffers():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()


class Meta(MetaModule):
    _ignore_buffers = {}
    _ignore_modules = {_BatchNorm}

    def __init__(self, model: nn.Module):
        super().__init__()
        from copy import deepcopy

        model = deepcopy(model)
        for k, v in model.__dict__.items():
            if isinstance(v, dict):
                self.__dict__.setdefault(k, dict()).update(v)
            elif isinstance(v, list):
                self.__dict__.setdefault(k, list()).extend(v)
            elif not callable(v):
                self.__dict__[k] = v
        self.__dict__["forward"] = model.forward

        Meta._move_params_to_buffer(self)


def meta(module: nn.Module):
    return Meta(module)


# pylint: disable=W0223
class MetaIterativeProblem(Problem):
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

        # grads
        grads = torch.autograd.grad(
            loss,
            self.trainable_parameters(),
            create_graph=not self._first_order,
            retain_graph=self._retain_graph,
            allow_unused=self._allow_unused,
        )

        # backward & zero grad & param update
        lr = self.optimizer.param_groups[0]["lr"]
        self.module.update_params(lr, grads)

        self.module.zero_grad()

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

                # patch module
                self.patch_module()

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

                self.module = self.module_orig

            self.ready = [False for _ in range(len(self._children))]

    def patch_module(self):
        """
        Patch PyTorch's native stateful module into the stateless module so as to support
        functional forward that takes params as its input.
        """
        self.module_orig = self.module
        self.module = meta(self.module)

    def cache_states(self):
        self.module_state_dict_cache = self.module.state_dict()
        if self.optimizer is not None:
            self.optimizer_state_dict_cache = self.optimizer.state_dict()

    def recover_states(self):
        self.module.load_state_dict(self.module_state_dict_cache)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(self.optimizer_state_dict_cache)

    def parameters(self):
        if hasattr(self.module, "params"):
            return list(self.module.params())
        else:
            return list(self.module.parameters())

    def trainable_parameters(self):
        if hasattr(self.module, "params"):
            return list(self.module.params())
        else:
            return list(self.module.parameters())

    def train(self):
        super().train()
        self.module.train()

    def eval(self):
        super().eval()
        self.module.eval()
