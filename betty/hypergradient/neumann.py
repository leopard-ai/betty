import torch

from betty.hypergradient.utils import add_with_none, neg_with_none


def neumann(loss, params, path, config, create_graph=True, retain_graph=False, allow_unused=True):
    """Approximate the matrix-vector multiplication with the best response Jacobian by the
    Neumann Series as proposed in
    `Optimizing Millions of Hyperparameters by Implicit Differentiation
    <https://arxiv.org/abs/1911.02590>`_ based on implicit function theorem (IFT). Users may
    specify learning rate (``neumann_alpha``) and unrolling steps (``neumann_iterations``) in
    ``Config``.

    :param loss: Outputs of the differentiated function.
    :type loss: `Tensor <https://pytorch.org/docs/stable/tensors.html#torch-tensor>`_
    :param params: Inputs with respect to which the gradient will be returned.
    :type params: Sequence of `Tensor <https://pytorch.org/docs/stable/tensors.html#torch-tensor>`_
    :param path: Path on which the gradient will be calculated.
    :type path: List of Problem
    :param config: Hyperparameters for the best-response Jacobian approximation
    :type config: Config
    :param create_graph:
        If ``True``, graph of the derivative will be constructed, allowing to compute higher order
        derivative products. Default: ``True``.
    :type create_graph: `bool <https://docs.python.org/3/library/functions.html#bool>`_, optional
    :param retain_graph:
        If ``False``, the graph used to compute the grad will be freed. Note that in nearly all
        cases setting this option to ``True`` is not needed and often can be worked around in a much
        more efficient way. Defaults to the value of ``create_graph``.
    :type retain_graph: `bool <https://docs.python.org/3/library/functions.html#bool>`_, optional
    :param allow_unused:
        If ``False``, specifying inputs that were not used when computing outputs (and therefore
        their grad is always zero) is an error. Defaults to ``False``.
    :type allow_unused: `bool <https://docs.python.org/3/library/functions.html#bool>`_, optional
    :return: The gradient of ``loss`` with respect to ``params``
    :rtype: List of `Tensor <https://pytorch.org/docs/stable/tensors.html#torch-tensor>`_
    """
    # direct grad
    direct_grad = torch.autograd.grad(loss,
                                      params,
                                      create_graph=create_graph,
                                      retain_graph=retain_graph,
                                      allow_unused=allow_unused)

    # implicit grad
    implicit_grad = torch.autograd.grad(loss,
                                        path[1].trainable_parameters(),
                                        retain_graph=False)
    for i in range(1, len(path)-1):
        implicit_grad = neumann_helper(implicit_grad, path[i], path[i+1], config)

    return [add_with_none(dg, ig) for dg, ig in zip(direct_grad, implicit_grad)]


def neumann_helper(vector, curr, prev, config):
    # ! Mabye replace with child.loss by adding self.loss attribute to save computation
    assert len(curr.paths) == 0, 'neumann method is not supported for higher order MLO!'
    in_loss = curr.training_step(curr.cur_batch)
    in_grad = torch.autograd.grad(in_loss, curr.trainable_parameters(), create_graph=True)
    v2 = approx_inverse_hvp(vector, in_grad, curr.trainable_parameters(),
                            iterations=config.neumann_iterations,
                            alpha=config.neumann_alpha)
    implicit_grad = torch.autograd.grad(in_grad, prev.trainable_parameters(), grad_outputs=v2)
    implicit_grad = [neg_with_none(ig) for ig in implicit_grad]

    return implicit_grad


def approx_inverse_hvp(v, f, params, iterations=3, alpha=1.):
    p = v
    for _ in range(iterations):
        hvp = torch.autograd.grad(f, params, grad_outputs=v, retain_graph=True)
        v = [v_i - alpha * hvp_i for v_i, hvp_i in zip(v, hvp)]
        p = [v_i + p_i for v_i, p_i in zip(v, p)]

    return [alpha * p_i for p_i in p]
