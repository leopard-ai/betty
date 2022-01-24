import torch
from itertools import repeat


class DifferentiableOptimizer:
    def __init__(self, loss_f, dim_mult, data_or_iter=None):
        """
        Args:
            loss_f: callable with signature (params, hparams, [data optional]) -> loss tensor
            data_or_iter: (x, y) or iterator over the data needed for loss_f
        """
        self.data_iterator = None
        if data_or_iter:
            self.data_iterator = data_or_iter if hasattr(data_or_iter, '__next__') else repeat(data_or_iter)

        self.loss_f = loss_f
        self.dim_mult = dim_mult
        self.curr_loss = None

    def get_opt_params(self, params):
        opt_params = [p for p in params]
        opt_params.extend([torch.zeros_like(p) for p in params for _ in range(self.dim_mult-1) ])
        return opt_params

    def step(self, params, hparams, create_graph):
        raise NotImplementedError

    def __call__(self, params, hparams, create_graph=True):
        with torch.enable_grad():
            return self.step(params, hparams, create_graph)

    def get_loss(self, params, hparams):
        if self.data_iterator:
            data = next(self.data_iterator)
            self.curr_loss = self.loss_f(params, hparams, data)
        else:
            self.curr_loss = self.loss_f(params, hparams)
        return self.curr_loss


class HeavyBall(DifferentiableOptimizer):
    def __init__(self, loss_f, step_size, momentum, data_or_iter=None):
        super(HeavyBall, self).__init__(loss_f, dim_mult=2, data_or_iter=data_or_iter)
        self.loss_f = loss_f
        self.step_size_f = step_size if callable(step_size) else lambda x: step_size
        self.momentum_f = momentum if callable(momentum) else lambda x: momentum

    def step(self, params, hparams, create_graph):
        n = len(params) // 2
        p, p_aux = params[:n], params[n:]
        loss = self.get_loss(p, hparams)
        sz, mu = self.step_size_f(hparams), self.momentum_f(hparams)
        p_new, p_new_aux = heavy_ball_step(p, p_aux, loss, sz,  mu, create_graph=create_graph)
        return [*p_new, *p_new_aux]


class Momentum(DifferentiableOptimizer):
    """
    GD with momentum step as implemented in torch.optim.SGD
    .. math::
              v_{t+1} = \mu * v_{t} + g_{t+1} \\
              p_{t+1} = p_{t} - lr * v_{t+1}
    """
    def __init__(self, loss_f, step_size, momentum, data_or_iter=None):
        super(Momentum, self).__init__(loss_f, dim_mult=2, data_or_iter=data_or_iter)
        self.loss_f = loss_f
        self.step_size_f = step_size if callable(step_size) else lambda x: step_size
        self.momentum_f = momentum if callable(momentum) else lambda x: momentum

    def step(self, params, hparams, create_graph):
        n = len(params) // 2
        p, p_aux = params[:n], params[n:]
        loss = self.get_loss(p, hparams)
        sz, mu = self.step_size_f(hparams), self.momentum_f(hparams)
        p_new, p_new_aux = torch_momentum_step(p, p_aux, loss, sz,  mu, create_graph=create_graph)
        return [*p_new, *p_new_aux]


class GradientDescent(DifferentiableOptimizer):
    def __init__(self, loss_f, step_size, data_or_iter=None):
        super(GradientDescent, self).__init__(loss_f, dim_mult=1, data_or_iter=data_or_iter)
        self.step_size_f = step_size if callable(step_size) else lambda x: step_size

    def step(self, params, hparams, create_graph):
        loss = self.get_loss(params, hparams)
        sz = self.step_size_f(hparams)
        return gd_step(params, loss, sz, create_graph=create_graph)


def gd_step(params, loss, step_size, create_graph=True):
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    return [w - step_size * g for w, g in zip(params, grads)]


def heavy_ball_step(params, aux_params, loss, step_size, momentum, create_graph=True):
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    return [w - step_size * g + momentum * (w - v) for g, w, v in zip(grads, params, aux_params)], params


def torch_momentum_step(params, aux_params, loss, step_size, momentum, create_graph=True):
    """
    GD with momentum step as implemented in torch.optim.SGD
    .. math::
              v_{t+1} = \mu * v_{t} + g_{t+1} \\
              p_{t+1} = p_{t} - lr * v_{t+1}
    """
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    new_aux_params = [momentum*v + g for v, g in zip(aux_params, grads)]
    return [w - step_size * nv for w, nv in zip(params, new_aux_params)], new_aux_params


