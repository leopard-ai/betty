Installation
============

Betty can be installed via `pip <https://pypi.org/project/pip/>`_. Betty's dependency includes:

- PyTorch 1.5 - 1.10
- Python 3.7 - 3.10
- Ubuntu 16.04 or later
- MacOS

Install with pip
~~~~~~~~~~~~~~~~

.. code::

  pip install betty

Verifying Installation
~~~~~~~~~~~~~~~~~~~~~~

You can verify the installation by running
`test.py <https://github.com/sangkeun00/betty/blob/main/examples/logistic_regression_hpo/test.py>`_.

.. code::

  python test.py

If the installation was successful, one should expect the output like:

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

  [YYYY-mm-dd HH:MM:SS] [INFO] [Problem "outer"] [Global Step 100] [Local Step 10] loss: 0.5861933827400208
  *** Welcome to BettyWorld ***

Hello (Betty) World!
