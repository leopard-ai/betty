Major Concepts
==============
In this chapter, we will introduce the basic concept of multilevel optimization (MLO), and
Betty's software design. Our design allows an easy-to-use, modular, and maintainable
programming interface for general MLO programs with complex dependencies without requiring expertise
in programming and mathematics. Particularly, this chapter is composed of 4 sub-chapters:

- :doc:`Multilevel Optimization <concept_mlo>`: We go through the mathematical formulation as well as several application examples of multilevel optimization.
- :doc:`Software Design <concept_software>`: We introduce our abstraction of multilevel optimization and how such abstraction is implemented within Betty. 
- :doc:`Autograd (Advanced) <concept_autograd>`: We discuss the mathematical underpinning of automatic differentiation for multilevel optimization based on the dataflow graph interpretation. We additionally illustrate how such automatic differentiation is implemented within Betty.
- :doc:`Architecture (Advanced) <concept_architecture>`: We discuss the overall software architecture of Betty and how it executes multilevel optimization.

.. toctree::
   :hidden:
   :maxdepth: 1

   Multilevel Optimization<concept_mlo.rst>
   Software Design<concept_software.rst>
   Autograd (Advanced)<concept_autograd.rst>
   Architecture (Advanced)<concept_architecture.rst>
