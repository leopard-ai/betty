Major Concepts
==============
In this chapter, we introduce the major concepts of multilevel optimization (MLO) and
Betty's software design. Betty allows for an easy-to-use, modular, and maintainable
programming interface for general MLO programs involving complex dependencies, without
requiring expertise in programming and mathematics. This chapter is composed of 4
sub-chapters:

- :doc:`Multilevel Optimization <concept_mlo>`: We go through the mathematical
  formulation as well as several application examples of multilevel optimization.
- :doc:`Software Design <concept_software>`: We introduce an abstraction for multilevel
  optimization and describe how this abstraction is implemented within Betty.
- :doc:`Autodiff (Advanced) <concept_autodiff>`: We discuss the mathematics of automatic
  differentiation for multilevel optimization. We then illustrate how automatic
  differentiation is implemented within Betty.
- :doc:`Architecture (Advanced) <concept_architecture>`: We discuss the overall software
  architecture of Betty and how it executes multilevel optimization.

.. toctree::
   :maxdepth: 1
   :hidden:

   Multilevel Optimization<concept_mlo.rst>
   Software Design<concept_software.rst>
   Autodiff (Advanced)<concept_autodiff.rst>
   Architecture (Advanced)<concept_architecture.rst>
