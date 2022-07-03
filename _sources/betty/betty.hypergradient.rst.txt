betty.hypergradient
===========================

Unlike traditional automatic differentiation techniques that calculate analytic Jacobian for each
operation, multilevel optimization requires *approximating* best-response Jacobian for each level
optimization problem. Below is the list of approximation techniques that are supported by Betty.

finite difference
-----------------
.. automodule:: betty.hypergradient.darts
   :members: darts
   :undoc-members:

neumann series
--------------
.. automodule:: betty.hypergradient.neumann
   :members: neumann
   :undoc-members:

conjugate gradient
------------------
.. automodule:: betty.hypergradient.cg
   :members: cg
   :undoc-members:

reinforce
---------
.. automodule:: betty.hypergradient.reinforce
   :members: reinforce
   :undoc-members:

