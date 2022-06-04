Tutorial
========

In this tutorial, we go through two major concepts --- **Problem** and **Engine** --- with the toy
example of hyperparameter optimization for logistic regression.

We first prepare train/validation data for logistic regression using ``numpy`` and ``torch``.

.. code:: python

    import import numpy as np
    from sklearn.model_selection import train_test_split
    import torch

    DATA_NUM = 1000
    DATA_DIM = 20

    w_gt = np.random.randn(DATA_DIM)
    x = np.random.randn(DATA_NUM, DATA_DIM)
    y = x @ w_gt + 0.1 * np.random.randn(DATA_NUM)
    y = (y > 0).astype(float)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.5)
    x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
    x_val, y_val = torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float()
