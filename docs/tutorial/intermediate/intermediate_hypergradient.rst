Hypergradient
=============

For researchers studying a novel hypergradient calculation method, we allow
a modular interface to implement their own algorithm without changing other
parts of the MLO program. Specifically, users can create a new python file
under the ``betty/hypergradient`` directory where they should implement their
own algorithm, and add a new algorithm to ``betty/hypergradient/__init__.py``
so that the ``Problem`` class can import and use it.

In the python file, users should define how matrix-vector multiplication
between best-response Jacobian of the current problem and the given vector
is calculated:

.. code:: python

    # betty/hypergradient/new_hypergrad.py
    def myhypergrad(vector, curr, prev):
        ...
        return matrix_vecotr_multiplication_with_best_response_Jacobian

Once users implement their own algorithm, in ``betty/hypergradient/__init__.py``,
they add their algorithm as follows:

.. code:: python

    from .new_hypergrad import myhypergrad

    jvp_fn_mapping = {"darts": darts, "neumann": neumann, "cg": cg, "reinforce": reinforce}
    jvp_fn_mapping['myhypergrad_name'] = myhypergrad