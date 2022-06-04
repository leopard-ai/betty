.. betty documentation master file, created by
   sphinx-quickstart on Wed Jun  1 12:29:13 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Betty Documentation
=================================

Betty is an automatic differentiation library for multilevel optimization.
Betty provides a PyTorch-like programming interface for users to write general
multilevel optimization code.

.. toctree::
   :caption: Quick Start
   :maxdepth: 2
   :hidden:

   Installation<quickstart/installation.rst>
   Tutorial<quickstart/tutorial.rst>
   Architecture (Advanced)<quickstart/architecture.rst>

.. toctree::
   :caption: Library Docs
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

   Neural Architecture Search<tutorial/neural_architecture_search.rst>
   Data Reweighting<tutorial/data_reweighting.rst>





.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
