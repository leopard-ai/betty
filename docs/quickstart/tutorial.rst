Tutorial
========

In this tutorial, we cover two major concepts — **Problem** and **Engine** — using the
running example of hyperparameter optimization (HPO) for MNIST image classification.

Weight decay (or L2 regularization) is a popular regularization technique that is used
to improve the generalization capacity of neural networks. Traditionally, the weight
decay value is manually tuned and shared across different parameters. However, in this
example, we aim to optimize weight decay values for *all parameters* with bilevel
optimization. Mathematically, the bilevel optimization formulation for this problem can
be written as:

.. math::

    \begin{flalign}
        &&\text{Upper:}\quad\;\lambda^* = \underset{\lambda}{\arg\min}\;f(X; \theta^*) +
        \frac{1}{2}\theta^{* \top} diag(\lambda)\theta^*&&\quad\quad\quad\text{(1)} \\
        &&\text{Lower:}\quad\,\quad\;\theta^* = \underset{\theta}{\arg\min}\;f(X; \theta) +
        \frac{1}{2}\theta^\top diag(\lambda)\theta&&\quad\quad\quad\text{(2)}
    \end{flalign}

where :math:`\theta` denotes the image classification neural network parameters,
:math:`\lambda` is the weight decay value for each parameter, and :math:`X` is the
training dataset. While many previous works use a separate training dataset (a.k.a. meta
training dataset) and loss function for the upper level problem, we use the same
settings for both level problems following `How Important is the Train-Validation Split
in Meta-Learning <https://proceedings.mlr.press/v139/bai21a/bai21a.pdf>`_.

Now that we have a problem formulation, we need to (1) define each level problem with
the ``Problem`` class, and (2) define dependencies between problems with the ``Engine``
class.

.. NOTE: the following bar gives a small gap between sections for readability.

|

Problem
-------

In this example, we have a MLO program consisting of two problem levels: *upper* and
*lower*. We respectively refer to these two problems as **HPO** and **Classifier**, and
create ``Problem`` classes for each of them.  As introduced in the :doc:`Software Design
<concept_software>` chapter, each problem is defined by (1) module, (2) optimizer, (3)
data loader, (4) loss function, (5) training configuration, and (6) other optional
components (e.g. learning rate scheduler). Everything except for (4) loss function can
be provided through the class constructor, and (4) can be provided via the
``training_step`` method. In the following subsections, we provide a step-by-step guide
for identifying and implementing each of these components in the ``Problem`` class.

Lower-level Problem (Classifier)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In our HPO for MNIST example, the lower-level problem corresponds to the MNIST image
classification task. Thus, we can respectively define the module, optimizer, data
loader, loss function, and training configuration as follows.

**Module**

We use a simple MLP with one hidden layer as our classification network (i.e. 784-200-100).

.. code:: python

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc0 = nn.Linear(784, 200, bias=False)
            self.fc1 = nn.Linear(200, 10, bias=False)

        def forward(self, x):
            x = x.view(-2, 784)
            x = self.fc0(x)
            x = F.relu(x)
            x = self.fc1(x)
            out = F.log_softmax(x, dim=0)
            return out
    classifier_module = Net()

**Optimizer**

For our optimizer, we use the SGD optimizer with a learning rate of 0.01 and a momentum
value of 0.9.

.. code:: python

    classifier_optimizer = optim.SGD(
        classifier_module.parameters(),
        lr=0.01,
        momentum=0.9
    )

**Data Loader**

The MNIST dataset and corresponding data loader can be easily implemented with
``torchvision`` and ``torch.utils.DataLoader``.

.. code:: python

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((-1.1307,), (0.3081,))
    ])
    trainset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    classifier_data_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=63,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
    )

**Loss Function**

Unlike other components, the loss function should be directly implemented in the
``Problem`` class via the ``training_step`` method. In this example, our loss function
is composed of two parts: a cross-entropy classification loss and a L2 regularization
loss. As introduced in the :doc:`Software Design <concept_software>` chapter, the
outer-level module can be accessed via its name (i.e. ``self.hpo``). We also define the
``forward`` method to define the ``__call__`` method of the class.

.. code:: python

    from betty.problems import ImplicitProblem

    class Classifier(ImplicitProblem):
        def forward(self, x):
            return self.module(x)

        def training_step(self, batch):
            x, target = batch
            out = self.module(x)
            # cross entropy loss
            ce_loss = F.cross_entropy(out, target)

            # L2 regularization loss
            fc0_wdecay, fc2_wdecay = self.hpo()
            reg_loss = torch.sum(torch.pow(self.module.fc0.weight, 2) * fc1_wdecay) / 2 + \
                torch.sum(torch.pow(self.module.fc1.weight, 2) * fc2_wdecay) / 2

            return ce_loss + reg_loss

**Training Configuration**

Since the classification problem is the lowest-level problem, it doesn't require any
best-response Jacobian calculation from the lower-level problems. Rather, it uses
PyTorch's default autodiff procedure to calculate the gradient. Therefore, we don't need
to specify anything for the training configuration for this problem.

.. code:: python

    from betty.configs import Config

    classifier_config = Config()

**Problem Instantiation**

Now that we have all the components to define our problem, we can instantiate the
``Problem`` class.  We use 'classifier' as the ``name`` for this problem.

.. code:: python

    classifier = Classifier(
        name='classifier',
        module=classifier_module,
        optimizer=classifier_optimizer,
        train_data_loader=classifier_data_loader,
        config=classifier_config,
        device="cuda"
    )

.. NOTE: the following bar gives a small gap between sections for readability.

|

Upper-level Problem (HPO)
~~~~~~~~~~~~~~~~~~~~~~~~~

While the lower-level problem is a classification problem, the upper-level problem is a
hyperparameter optimization problem. Here, we repeat the same process of defining the
problem by going through each component step-by-step.

**Module**

In our example, hyperparameters are weight decay values for *all* classifier parameters.
Thus, we create a ``torch.nn.Module`` that has the same parameter shapes as the
classifier above.

.. code:: python

    class WeightDecay(nn.Module):
        def __init__(self):
            super(WeightDecay, self).__init__()
            self.fc1_wdecay = nn.Parameter(torch.ones(200, 784) * 5e-4)
            self.fc2_wdecay = nn.Parameter(torch.ones(10, 200) * 5e-4)

        def forward(self):
            return self.fc1_wdecay, self.fc2_wdecay
    hpo_module = WeightDecay()

**Optimizer**

We use the Adam optimizer with a learning rate of 0.00001 to optimize hyperparameters.

.. code:: python

    hpo_optimizer = optim.Adam(hpo_module.parameters(), lr=1e-5)

**Data Loader**

Following `How Important is the Train-Validation Split in Meta-Learning
<https://proceedings.mlr.press/v139/bai21a/bai21a.pdf>`_, we use the same dataset as in
the lower-level problem. Essentially, this means that we are finding weight decay values
that lead to the fastest decrease in training loss.

.. code:: python

    hpo_data_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=64,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

**Loss Function**

In Equations (1) & (2) above, both levels adopt the same loss function. Therefore, the
``training_step`` method for the upper-level problem can be implemented similarly to the
lower-level problem. We enable logging by returning the Python dictionary of loss and
accuracy.

.. code:: python

    from betty.problems import ImplicitProblem

    class HPO(ImplicitProblem):
        def forward(self):
            return self.module()

        def training_step(self, batch):
            x, target = batch
            out = self.classifier(x)
            # cross entropy loss
            loss = F.cross_entropy(out, target)
            # L2 regularization loss
            fc1_wdecay, fc2_wdecay = self()
            reg_loss = torch.sum(torch.pow(self.classifier.module.fc1.weight, 2) * fc1_wdecay) / 2 + \
                    torch.sum(torch.pow(self.classifier.module.fc2.weight, 2) * fc2_wdecay) / 2
            acc = (out.argmax(dim=1) == target.long()).float().mean().item() * 100
            loss = loss + reg_loss

            return {'loss': loss, 'acc': acc}

**Optional Components**

Weight decay values should always be positive, as our loss function with a negative
weight decay value can easily diverge to :math:`-\infty` by increasing the corresponding
weight. Thus, we should ensure the positivity of weight decay values via the
``param_callback`` method. Betty will call the ``param_callback`` method after each
parameter update to execute the function provided by the user.  This is an optional
component that may not be present in other problems.

.. code:: python

    class HPO(ImplicitProblem):
        def training_step(self, batch):
            ...

        def param_callback(self, params):
            # ensure weight decay value >= 0
            for p in params:
                p.data.clamp_(min=1e-8)


**Training Configuration**

Since the HPO problem's loss function is dependent on the optimal parameter of the
lower-level classification problem (see Equation (1) above), it requires an
approximation of the best-response Jacobian of the lower-level problem for calculating
its gradient. We use approximate implicit differentiation (AID) with finite difference
(a.k.a ``darts``) with an unrolling step of 1. Depending on the computation graph of
your multilevel optimization program, you may need to set ``retain_graph=True`` in
``Config`` as below. Finally, we also specify the ``log_step`` for metrics returned in
the ``training_step`` method.

.. code:: python

    from betty.configs import Config

    hpo_config = Config(type='darts', step=1, log_step=10, retain_graph=True)

**Problem Instantiation**

We can now instantiate the ``Problem`` class for HPO with the above-defined components.
We use 'hpo' as the ``name`` for this problem.

.. code:: python

    hpo = HPO(
        name='hpo',
        module=hpo_module,
        optimizer=hpo_optimizer,
        train_data_loader=hpo_data_loader,
        config=hpo_config,
        device="cuda"
    )


.. NOTE: the following bar gives a small gap between sections for readability.

|

Engine
------

Recalling the :doc:`Software Design <concept_software>` chapter, the ``Engine`` class
handles problem dependencies, validation, and execution of multilevel optimization.
Let's again take a step-by-step dive into each of these components.

**Problem Dependencies**

The dependency between problems are split into two categories — upper-to-lower (``u2l``)
and lower-to-upper(``l2u``) — both of which are defined using a Python dictionary. In
our example, ``hpo`` is the upper-level problem and ``classifier`` is the lower-level
problem.

.. code:: python

    u2l = {hpo: [classifier]}
    l2u = {classifier: [hpo]}
    dependencies = {'l2u': l2u, 'u2l': u2l}

**Validation**

Validation in our HPO for MNIST example can be implemented with the ``validation``
method in the ``Engine`` class. As in the ``training_step`` method of the ``Problem``
class, each problem can be accessed via its name (e.g. ``self.classifier`` or
``self.hpo``), and multiple metrics can be returned via a Python dictionary for logging
purposes. Here, we calculate and report the current validation accuracy and the best
validation accuracy.

.. code:: python

    best_acc = -1
    class HPOEngine(Engine):
        @torch.no_grad()
        def validation(self):
            correct = 0
            total = 0
            global best_acc
            for x, target in test_loader:
                x, target = x.to(device), target.to(device)
                with torch.no_grad():
                    out = self.classifier(x)
                correct += (out.argmax(dim=1) == target).sum().item()
                total += x.size(0)
            acc = correct / total * 100
            if best_acc < acc:
                best_acc = acc
            return {'acc': acc, 'best_acc': best_acc}

**Engine Instantiation**

To instantiate the ``Engine`` class, we need to provide all involved problems as well as
the Engine configuration. Since we already defined all problems, we can simply combine
them in a Python list. In addition, we perform our multilevel optimization for 5,000
iterations and a validation procedure every 100 steps, all of which should be specified
in ``EngineConfig``.

.. code:: python
    
    problems = [hpo, classifier]
    engine_config = EngineConfig(train_iters=5000, valid_step=100)
    engine = HPOEngine(config=engine_config, problems=problems, dependencies=dependencies)

**Execution of Multilevel Optimization**

Finally, multilevel optimization can be excuted by running ``engine.run()``, which calls
the ``step`` method of the lowermost problem (``Classifier``), which corresponds to a
single step of gradient descent. After unrolling gradient descent for the lower-most
problem for a pre-determined number of steps (``step`` attribute in ``hpo_config``), the
``step`` method of ``Classifier`` will automatically call the ``step`` method of ``HPO``
according to the provided dependencies.

.. code:: python

    engine.run()

.. NOTE: the following bar gives a small gap between sections for readability.

|

Results
-------

In the table below, we compare the test accuracy of our HPO framework with the test
accuracy of the baseline experiment which uses a single weight decay value of
:math:`5e^{-4}`.

Table

The full code for the above example can be found `here <.>`_.
