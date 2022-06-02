import torch

from betty.hypergradient.utils import add_with_none, neg_with_none


def reinforce(loss, params, path, config, create_graph=True, retain_graph=False, allow_unused=True):
    """Approximate the matrix-vector multiplication with the best response Jacobian by the
    REINFORCE method. The use of REINFORCE algorithm allows users to differentiate through
    optimization with non-differentiable processes such as sampling. This method has not been
    completely implemented yet.

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
    # TODO: recursion
    #for i in range(1, len(path)-1):
    #    implicit_grad = darts_helper(implicit_grad, path[i], path[i+1], config)

    return [add_with_none(dg, ig) for dg, ig in zip(direct_grad, implicit_grad)]


def reinforce_helper(vector, curr, prev, config):
    raise NotImplementedError