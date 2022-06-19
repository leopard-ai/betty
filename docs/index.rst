Betty Documentation
===================

Introduction
------------
Betty is a `PyTorch <https://pytorch.org>`_ library for multilevel optimization (MLO) and/or
generalized meta-learning that provides a unified programming interface for a number of
MLO applications including meta-learning, hyperparameter optimization, neural architecture
search, data reweighting, adversarial learning, and reinforcement learning.

Benefits
--------
Implementing multilevel optimization is notoriously complicated. For example, it
requires approximating gradients using iterative/implicit differentiation, and writing
nested for-loops to handle hierarchical dependencies between multiple levels.

Betty aims to abstract away low-level implementation details behind its API, while
allowing users to write only high-level declarative code. Now, users simply need to do
two things to implement any MLO program:

1. Define each level's optimization problem using the **Problem** class.
2. Define the hierarchical problem structure using the **Engine** class.

From here, Betty performs automatic differentiation for the MLO program,
choosing from a set of provided gradient approximation methods, in order to carry out
robust, high-performance MLO.

Applications
------------
Betty can be used for implementing a wide range of MLO applications including
hyperparameter optimization, neural architecture serach, data reweighting, etc. We
provide several implementation examples for:

- `Hyperparameter Optimization <examples/logistic_regression_hpo/>`_
- `Neural Architecture Search <examples/neural_architecture_search/>`_
- `Data Reweighting <examples/learning_to_reweight/>`_
- `Domain Adaptation for Pretraining & Finetuning <examples/learning_by_ignoring/>`_
- `(Implicit) Model-Agnostic Meta-Learning <examples/maml/>`_ (Under development)
 
While each of above examples traditionally have distinct implementation styles, one
should notice that our implementation shares the same code structure thanks to Betty.
We plan to implement more MLO applications in the future.

Getting Started
---------------

- :doc:`Installation <quickstart/installation>`
- :doc:`Tutorial <quickstart/tutorial>`

.. toctree::
   :caption: Get Started
   :maxdepth: 2
   :hidden:

   Installation<quickstart/installation.rst>
   Major Concepts<quickstart/concept.rst>

.. toctree::
   :caption: Tutorial
   :maxdepth: 2
   :hidden:

   Basic<tutorial/basic/basic.rst>
   Intermediate<tutorial/intermediate/intermediate.rst>

.. toctree::
   :caption: API Docs
   :maxdepth: 2
   :hidden:

   betty.engine<betty/betty.engine.rst>
   betty.problems<betty/betty.problems.rst>
   betty.hypergradient<betty/betty.hypergradient.rst>
   betty.optim<betty/betty.optim.rst>
   betty.logging<betty/betty.logging.rst>

.. toctree::
   :caption: Examples
   :hidden:

   Neural Architecture Search<examples/neural_architecture_search.rst>
   Data Reweighting<examples/data_reweighting.rst>



.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
