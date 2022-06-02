import torch


def default(loss, params, path, config, create_graph=True, retain_graph=False, allow_unused=False):
    """Approximate the matrix-vector multiplication with the best response Jacobian by the
    (PyTorch's) default autograd method. This method has been widely popularized by
    `Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (MAML)
    <https://arxiv.org/abs/1703.03400>`_.

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
    grad = torch.autograd.grad(
        loss,
        params,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=allow_unused
    )
    return grad
