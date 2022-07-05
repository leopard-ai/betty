Betty Documentation
===================

Introduction
------------
Betty is a `PyTorch <https://pytorch.org>`_ library for generalized-meta learning (GML)
and multilevel optimization (MLO) that provides a unified programming interface for a
number of GML/MLO applications including meta-learning, hyperparameter optimization,
neural architecture search, data reweighting, adversarial learning, and reinforcement
learning.

Below is the figure that illustrates the concept of GML/MLO.

.. figure:: _static/imgs/mlo.png
    :align: center

    Figure. Visual illustration of the concept of GML/MLO.

Benefits
--------
Implementing generalized meta-learning and multilevel optimization is notoriously
complicated. For example, it requires approximating gradients using
iterative/implicit differentiation, and writing nested for-loops to handle
hierarchical dependencies between multiple levels.

Betty aims to abstract away low-level implementation details behind its API, while
allowing users to write only high-level declarative code. Now, users simply need to do
two things to implement any GML/MLO program:

1. Define each level's optimization problem using the **Problem** class.
2. Define the hierarchical problem structure using the **Engine** class.

From here, Betty performs automatic differentiation for the MLO program,
choosing from a set of provided gradient approximation methods, in order to carry out
robust, high-performance GML/MLO.

Applications
------------
Betty can be used for implementing a wide range of GML/MLO applications including
hyperparameter optimization, neural architecture serach, data reweighting, etc. We
provide several reference implementation examples for:

- `Hyperparameter Optimization <https://github.com/leopard-ai/betty/tree/main/examples/logistic_regression_hpo>`_
- `Neural Architecture Search <https://github.com/leopard-ai/betty/tree/main/examples/neural_architecture_search>`_
- `Data Reweighting <https://github.com/leopard-ai/betty/tree/main/examples/learning_to_reweight>`_
- `Domain Adaptation for Pretraining & Finetuning <https://github.com/leopard-ai/betty/tree/main/examples/learning_by_ignoring>`_
- `(Implicit) Model-Agnostic Meta-Learning <https://github.com/leopard-ai/betty/tree/main/examples/implicit_maml>`_
 
While each of above examples traditionally have distinct implementation styles, Betty
allows for common code structures and autodiff routines to be used in all examples. We
plan to implement more GML/MLO applications in the future.

Getting Started
---------------

- :doc:`Installation <quickstart/installation>`
- :doc:`Tutorials <tutorial/basic/basic>`

.. toctree::
   :hidden:
   :caption: Get Started
   :maxdepth: 1

   Installation<quickstart/installation.rst>
   Major Concepts<quickstart/concept.rst>

.. toctree::
   :hidden:
   :caption: Tutorials
   :maxdepth: 2

   Basic<tutorial/basic/basic.rst>
   Intermediate<tutorial/intermediate/intermediate.rst>

.. toctree::
   :hidden:
   :caption: API Docs
   :maxdepth: 2

   betty.engine<betty/betty.engine.rst>
   betty.problems<betty/betty.problems.rst>
   betty.hypergradient<betty/betty.hypergradient.rst>
   betty.optim<betty/betty.optim.rst>
   betty.logging<betty/betty.logging.rst>

.. toctree::
   :hidden:
   :caption: Examples
   :maxdepth: 1

   Data Reweighting<examples/data_reweighting.rst>
   Neural Architecture Search<examples/neural_architecture_search.rst>



.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
