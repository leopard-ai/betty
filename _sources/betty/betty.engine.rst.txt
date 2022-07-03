betty.engine
===========================

Unlike traditional automatic differentiation techniques that calculate analytic Jacobian for each
operation, multilevel optimization requires *approximating* best-response Jacobian for each level
optimization problem. Below is the list of approximation techniques that are supported by Betty.

Engine
------
.. automodule:: betty.engine
   :members:
   :exclude-members: training_step, dfs
   :undoc-members: