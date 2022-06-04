Major Concepts
==============

Betty allows an easy-to-use, modular, and maintainable programming interface for multilevel
optimization (MLO) by breaking down MLO into two high-level concepts --- (1) Optimization Problems,
and (2) Problem Dependencies --- for which we design abstract Python classes:

- ``Problem`` class: Abstraction of optimization problems
- ``Engine`` class: Abstraction of problem dependencies

In this chapter, we will introduce each of these concepts/classes in depth.

Problem
-------

Each optimization problem :math:`P_k`` in MLO is defined by the trainable module
(e.g. torch.nn.Module), the sets of the upper and lower constraining problems
:math:`\mathcal{U}_k\;\&\;\mathcal{L}_k`, the dataset :math:`\mathcal{D}_k`, the cost function
:math:`\mathcal{C}_k`, the optimizer, and other optimization configurations (e.g. best-response
Jacobian calculation algorithm, number of unrolling steps). The ``Problem`` class is an interface
where users can provide each of the aforementioned components, except for the constraining problem
sets, to define the optimization problem. We intentionally use a design where the constraining
problem sets are provided, rather, by the ``Engine`` class. In doing so, users need to provide the
hierarchical problem dependencies only once when they initialize ``Engine``, and can avoid the
potentially error-prone and cumbersome process of providing constraining problems manually every
time they define new problems. As for the remaining components, each one except for the cost
function :math:`\mathcal{C}_k`` can be provided through the class constructor, while the cost
function can be defined through a ``training_step`` method. The example usage of ``Problem`` is
shown below:

.. code:: python

    class MyProblem(Problem):
        def training_step(self, batch):
            # Users define the cost function here
            return cost_fn(batch, self.module, self.other_probs, ...)
        
    config = Config(type="darts", steps=5, first_order=True, retain_graph=True)
    prob = MyProblem(name, config, module, optimizer, data_loader)

Importantly, we provide a modular interface for users to choose different best-response Jacobian
algorithms via a one-liner change in ``Config``. This allows users without mathematical and
programming expertise to easily and flexibly write a MLO code. All in all, our ``Problem`` and its
``step`` method are similar in concept to PyTorch's ``nn.Module`` and its ``forward`` method.


Engine
------
