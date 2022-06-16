betty.optim
===================

Iterative differentiation (ITD) differentiates through the optimization path, requiring 
tracking of the intermediate states of the parameter during optimization. However, native
PyTorch optimizer overrides such intermediate states with the use of in-place operations for the
good purpose of saving memory. We here provide the functionality of patching PyTorch's native
optimizers by substituting all involved in-place operations to allow ITD. When users pass
their optimizer to ``IterativeProblem`` class through the constructor, ``initialize`` method of
``IterativeProblem`` will automatically call ``patch_optimizer`` and ``patch_scheduler`` to patch
user-proviced PyTorch native optimizer/scheduler along with stateful modules.

.. automodule:: betty.optim
   :members:
   :undoc-members:

Supported optimizers
--------------------
Below is the list of differentiable optimizers supported by Betty.

.. automodule:: betty.optim.sgd
   :members:
   :exclude-members: step
   :undoc-members:

.. automodule:: betty.optim.adam
   :members:
   :exclude-members: step
   :undoc-members:

.. automodule:: betty.optim.adamw
   :members:
   :exclude-members: step
   :undoc-members:
