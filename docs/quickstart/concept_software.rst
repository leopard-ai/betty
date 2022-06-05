Software Design
===============

Betty allows an easy-to-use, modular, and maintainable programming interface for multilevel
optimization (MLO) by breaking down MLO into two high-level concepts --- (1) Optimization Problems,
and (2) Problem Dependencies --- for which we design abstract Python classes:

- ``Problem`` class: Abstraction of optimization problems
- ``Engine`` class: Abstraction of problem dependencies

In this chapter, we will introduce each of these concepts/classes in depth.

Problem
-------

Each optimization problem :math:`P` in MLO is defined by the trainable module
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
algorithms via a one-liner change in ``Config``. 


Engine
------

While ``Problem`` manages each optimization problem, ``Engine`` handles a dataflow graph based on
the user-provided hierarchical problem dependencies. As discussed in the previous
`Chapter <.>`_, a dataflow graph for MLO has upper-to-lower and
lower-to-upper directed edges. We allow users to define two separate graphs, one for each type
of edge, using a Python dictionary, in which keys/values respectively represent start/end nodes
of the edge. When user-defined dependency graphs are provided, ``Engine`` compiles them and
finds all paths required for automatic differentiation with a modified depth-first search algorithm.
Moreover, ``Engine`` sets constraining problem sets for each problem based on the dependency graphs,
as mentioned above. Once all initialization processes are done, users can run a whole MLO program by
calling ``Engine``'s run method, which repeatedly calls ``step`` methods of lowermost problems. 
The example usage of ``Engine`` is provided below:

.. code:: python

    prob1 = MyProblem1(...)
    prob2 = MyProblem2(...)
    depend = {"u2l": {prob1: [prob2]}, "l2u": {prob1: [prob2]}}
    engine = Engine(problems=[prob1, prob2], dependencies=depend)
    engine.run()

To summarize, Betty provides a PyTorch-like programming interface of defining multiple optimization
problems, which can scale up to large MLO programs with complex dependencies, as well as a modular
interface for a variety of best-response Jacobian algorithms, without requiring mathematical and
programming proficiency.
