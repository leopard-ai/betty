Installation
============

Betty can be installed via `pip <https://pypi.org/project/pip/>`_. Betty's dependencies include:

- Python 3.6 - 3.9
- PyTorch 1.6 - 1.12

Install with pip
~~~~~~~~~~~~~~~~

.. code::

  pip install betty-ml

To use ``IterativeProblem``, you need to additionally install ``functorch``. However,
``functorch`` requires users to use PyTorch 1.11.

.. code::

  pip install functorch

To install Betty and develop locally:

.. code::

  git clone https://github.com/leopard-ai/betty.git
  cd betty
  pip install -e .

Verifying Installation
~~~~~~~~~~~~~~~~~~~~~~

You can verify the installation by running
`test.py <https://github.com/leopard-ai/betty/blob/main/examples/logistic_regression_hpo/test.py>`_.

.. code::

  python test.py

If the installation was successful, you should see the following output:

.. code::

  [YYYY-mm-dd HH:MM:SS] [INFO] Initializing Multilevel Optimization...

  [YYYY-mm-dd HH:MM:SS] [INFO] *** Problem Information ***
  [YYYY-mm-dd HH:MM:SS] [INFO] Name: outer
  [YYYY-mm-dd HH:MM:SS] [INFO] Uppers: []
  [YYYY-mm-dd HH:MM:SS] [INFO] Lowers: ['inner']
  [YYYY-mm-dd HH:MM:SS] [INFO] Paths: [['outer', 'inner', 'outer']]

  [YYYY-mm-dd HH:MM:SS] [INFO] *** Problem Information ***
  [YYYY-mm-dd HH:MM:SS] [INFO] Name: inner
  [YYYY-mm-dd HH:MM:SS] [INFO] Uppers: ['outer']
  [YYYY-mm-dd HH:MM:SS] [INFO] Lowers: []
  [YYYY-mm-dd HH:MM:SS] [INFO] Paths: []

  [YYYY-mm-dd HH:MM:SS] [INFO] Time spent on initialization: 0.001 (s)

  [YYYY-mm-dd HH:MM:SS] [INFO] [Problem "outer"] [Global Step 1000] [Local Step 10] loss: 0.3682613968849182
  [YYYY-mm-dd HH:MM:SS] [INFO] [Problem "outer"] [Global Step 2000] [Local Step 20] loss: 0.30229413509368896
  [YYYY-mm-dd HH:MM:SS] [INFO] [Problem "outer"] [Global Step 3000] [Local Step 30] loss: 0.29078295826911926
  [YYYY-mm-dd HH:MM:SS] [INFO] [Problem "outer"] [Global Step 4000] [Local Step 40] loss: 0.29050588607788086
  [YYYY-mm-dd HH:MM:SS] [INFO] [Problem "outer"] [Global Step 5000] [Local Step 50] loss: 0.29037463665008545
  *** Hello (Betty) World ***

Hello (Betty) World!
