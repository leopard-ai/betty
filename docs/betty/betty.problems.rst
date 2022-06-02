betty.problems
======================

Problem
-------

.. automodule:: betty.problems.problem
   :members: 
   :exclude-members: __call__
   :undoc-members:

Implicit Problem
----------------

Implicit problem is used when best-response Jacobian for the current problem is calculated with
(approximate) implicit differentiation (AID). AID doesn't require patching module/optimizer, and
usually achieve memory & compute efficiency especially when the large unrolling step is used. We
recommend users to use ``ImplicitProblem`` as a default class to define problems in MLO.

.. automodule:: betty.problems.implicit_problem
   :members:
   :exclude-members:
      train, eval, trainable_parameters, cache_states, recover_states, optimizer_step, parameters
   :undoc-members:

Iterative Problem
-----------------

Iterative Problem is used when best-response Jacobian for the current problem is calculated with
iterative differentiation (ITD). ITD requires patching module/optimizer to track intermediate
parameter states for the gradient flow. We discourage users to use this class, because
memory/computation efficiency of ITD is oftentimes worse than AID. In addition, users may need to
be familiar with functional programming style due to the use of stateless modules.

.. automodule:: betty.problems.iterative_problem
   :members:
   :exclude-members:
      train, eval, trainable_parameters, cache_states, recover_states, optimizer_step, parameters,
      initialize, initialize_optimizer_state
   :undoc-members:


