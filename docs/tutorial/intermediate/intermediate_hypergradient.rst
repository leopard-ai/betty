Hypergradient
=============

For researchers studying a novel hypergradient calculation method, we allow
a modular interface to implement their own algorithm without changing other
parts of the MLO program. Specifically, users can create a new python file
under the ``betty/hypergradient`` directory where they should implement their
own algorithm, and add a new algorithm to ``betty/hypergradient/__init__.py``
so that the ``Problem`` class can import and use it.

Recall that automatic differentiation for MLO is achieved by iteratively
performing matrix-vector mulitplication with best-response Jacobian (See
:doc:`../../quickstart/concept_autodiff`):

.. math::

    \begin{flalign}
        &&\text{Calculate}\,:\quad&\frac{\partial w^*(\lambda)}{\partial \lambda}\times v\\[2ex]
        &&\text{Given}\,:\quad&w^*(\lambda) = \underset{w}{\mathrm{argmin}}\;\mathcal{C}(w, \lambda)
    \end{flalign}

In the python file, users should define how above "Calculate" part would be
performed given :math:`\lambda, w,\text{ and } v`. 


.. code:: python

    # betty/hypergradient/new_hypergrad.py
    def myhypergrad(vector, curr, prev):
        """
        vector: corresponds to v
        curr: corresponds to the lower-level problem whose parameter is w
        prev: corresponds to the upper-level problem whose parameter is \lambda
        """
        ...
        return matrix_vecotr_multiplication_with_best_response_Jacobian

Once users implement their own algorithm, in ``betty/hypergradient/__init__.py``,
they add their algorithm as follows:

.. code:: python

    from .new_hypergrad import myhypergrad

    jvp_fn_mapping = {"darts": darts, "neumann": neumann, "cg": cg, "reinforce": reinforce}
    jvp_fn_mapping['myhypergrad_name'] = myhypergrad