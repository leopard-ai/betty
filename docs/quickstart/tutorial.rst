Tutorial
========

In this tutorial, we go through two major concepts --- **Problem** and **Engine** --- with the
example of hyperparameter optimization (HPO) for MNIST image classification.
Weight decay (or L2 regularization) is a popular regularization tehcnique that is used to improve
the generalization capacity of neural networks.
While the weight decay value is manually tuned and shared across different parameters traditionally,
we aim to optimize weight decay values for *all parameters* with bilevel optimization in this
example. 
Mathematically, the bilevel optimization formulation for the above problem can be written as:

.. math::

    \text{Upper:}\quad\;\lambda^* = \underset{\lambda}{\arg\min}\;f(X; \theta^*) +
    \frac{1}{2}\theta^{* \top} diag(\lambda)\theta^* \\
    \text{Lower:}\quad\,\quad\;\theta^* = \underset{\theta}{\arg\min}\;f(X; \theta) +
    \frac{1}{2}\theta^\top diag(\lambda)\theta

where :math:`\theta` is the image classification network parameter, :math:`\lambda` is the weight 
decay value for each parameter, and :math:`X` is the training dataset. While many previous work use
separate training dataset (a.k.a meta training dataset) and loss function for the upper level
problem, we use the same settings for both level problems following
`How Important is the Train-Validation Split in Meta-Learning
<https://proceedings.mlr.press/v139/bai21a/bai21a.pdf>`_.

Now that we have a problem formulation, we need to (1) define each level problem with the 
``Problem`` class, and (2) define dependency between problems with the ``Engine`` class.

Problem
-------
In this example, we have two levels of problems. We respectively refer to upper- and lower-level
problems as **HPO** and **Classifier**, and create ``Problem`` classes for each of them.

Upper-level Problem (HPO)
~~~~~~~~~~~~~~~~~~~~~~~~~
As introduced in this `Chapter <.>`_, each problem is defined by (1) module, (2) optimizer,
(3) data loader, (4) loss function, (5) training configurations, and (6) other optional components
(e.g. learning rate scheduler), all of which can be provided via the ``Problem`` class constructor.

.. code:: python

    # (1) module
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(784, 200, bias=False)
            self.fc2 = nn.Linear(200, 10, bias=False)

        def forward(self, x):
            x = x.view(-1, 784)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            out = F.log_softmax(x, dim=1)
            return out
    net = Net()

    # (2) optimizer


    # (3) data loader

    # (4) loss function



Lower-level Problem (Classifier)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

