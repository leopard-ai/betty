Quick Start
===========

Throughout our tutorials, we will use **Data Reweighting for Long-Tailed Image
Classification** as our running example.  The basic context is that we aim to mitigate a
class imbalance problem (or long-tailed distribution problem) by re-assigning
higher/lower weights to data from rare/common classes. In particular, `Meta-Weight-Net
(MWN) <https://arxiv.org/abs/1902.07379>`_ proposes to approach data reweighting with
bilevel optimization as follows:

.. math::

        w^*=\underset{w}{\mathrm{argmin}}\;\mathcal{L}_{rwt}(\theta^*(w))\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\;\;\,\text{Reweighting}\\
        \text{s.t. }\theta^*(w)=\underset{\theta}{\mathrm{argmin}}\;\frac{1}{N}\sum_{i=1}^n\mathcal{R}(L^i_{cls}(\theta);w)\cdot L^i_{cls}(\theta)\quad\quad\quad\text{Classification}

where :math:`\theta` is the classifier network parameters, :math:`L_{cls}^i` is the
classification loss (cross-entropy) for the :math:`i`-th training sample,
:math:`\mathcal{L}_{rwt}` is the loss for the reweighting level (cross-entropy) and
:math:`w` is the MWN :math:`\mathcal{R}`'s parameters, which reweights each training
sample given its training loss :math:`L^i_{train}`.

Now that we have a problem formulation, we need to (1) define each level problem with
the ``Problem`` class, and (2) define dependencies between problems with the ``Engine``
class.

.. NOTE: the following bar gives a small gap between sections for readability.

|

Problem
-------
In this example, we have a MLO program consisting of two problem levels: *upper* and
*lower*. We respectively refer to these two problems as **Reweight** and **Classifier**,
and create ``Problem`` classes for each of them.  As introduced in the
:doc:`Software Design <../../quickstart/concept_software>` chapter, each problem is defined
by (1) module, (2) optimizer, (3) data loader, (4) loss function, (5) training configuration,
and (6) other optional components (e.g. learning rate scheduler). Everything except for (4)
loss function can be provided through the class constructor, and (4) can be provided via the
``training_step`` method. In the following subsections, we provide a step-by-step guide
for identifying and implementing each of these components in the ``Problem`` class.

Lower-level Problem (Classifier)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In our data reweighting example, the lower-level problem corresponds to the long-tailed
CIFAR image classification task. Specifically, we set the imbalance factor to 50, meaning
that the most common class has 50 times more data than the least common class. The data
loader code is adopted from
`here
<https://github.com/ShiYunyi/Meta-Weight-Net_Code-Optimization/blob/main/noisy_long_tail_CIFAR.py>`_.
We can respectively define the module, optimizer, data loader, loss function, and training
configuration as follows.

**Module, Optimizer, Data Loader, (optional) Scheduler**

.. code:: python

    # Module
    classifier_module = ResNet32(num_classes=10)

    # Optimizer
    classifier_optimizer = optim.SGD(classifier_module.parameters(),
                                     lr=0.1,
                                     momentum=0.9,
                                     weight_decay=5e-4)

    # Data Loader
    classifier_dataloader, *_ = build_dataloader(dataset="cifar10",
                                                 imbalanced_factor=50,
                                                 batch_size=100)

    # LR Scheduler
    classifier_scheduler = optim.lr_scheduler.MultiStepLR(classifier_optimizer,
                                                          milestones=[5000, 7500, 9000],
                                                          gamma=0.1)

**Loss Function**

Unlike other components, the loss function should be directly implemented in the
``Problem`` class via the ``training_step`` method.

.. code:: python

    from betty.problems import ImplicitProblem

    class Classifier(ImplicitProblem):
        def training_step(self, batch):
            inputs, labels = batch
            outputs = self.forward(inputs)
            loss_vector = F.cross_entropy(outputs, labels.long(), reduction="none")

            # Reweight
            loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))
            weight = self.reweight(loss_vector_reshape.detach())
            loss = torch.mean(weight * loss_vector_reshape)

            return loss

In this example, we aim to overcome a long-tailed distribution by reweighting each data
sample (e.g. increasing weights for data from rare classes while decreasing weights for
data from common classes). This is achieved by interacting with the upper-level
**Reweight** problem. The Engine class will provide an access to the **Reweight** problem
via its name for the **Classifier** problem (i.e.
:code:`weight = self.reweight(loss_vector_reshape.detach())`). Thus, users should be
aware of names of other problems, with which the current problem interacts, when
writing the loss function.

**Training Configuration**

Since the **Classifier** problem is the lowest-level problem, it doesn't require any
best-response Jacobian calculation from the lower-level problems. Rather, it uses
PyTorch's default autodiff procedure to calculate the gradient. Therefore, we don't need
to specify anything for the training configuration for this problem.

.. code:: python

    from betty.configs import Config

    classifier_config = Config()

**Problem Instantiation**

Now that we have all the components to define the **Classifier** problem, we can
instantiate the ``Problem`` class.  We use 'classifier' as the ``name`` for this
problem.

.. code:: python

    classifier = Classifier(
        name='classifier',
        module=classifier_module,
        optimizer=classifier_optimizer,
        scheduler=classifier_scheduler,
        train_data_loader=classifier_dataloader,
        config=classifier_config,
        device="cuda"
    )

|

Upper-level Problem (Reweight)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While the lower-level problem is a classification problem, the upper-level problem is a
reweighting problem. Specifically,
`Meta-Weight-Net (MWN) <https://arxiv.org/abs/1902.07379>`_ proposes to reweight each
data sample with one hidden layer MLP that takes a loss value as an input and outputs an
importance weight. 

**Module, Optimizer, Data Loader**

.. code:: python

    # Module
    class MLP(nn.Module):
        def __init__(self, hidden_size=100):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(1, hidden_size)
            self.fc2 = nn.Linear(hidden_size, 1)

        def forward(self, x):
            x = self.fc2(F.relu(self.fc1(x)))
            weight = torch.sigmoid(x)

            return weight

    reweight_module = MLP(hidden_size=100)

    # Optimizer
    reweight_optimizer = optim.Adam(reweight_module.parameters(), lr=1e-5)
    
    # Data Loader
    _, reweight_dataloader, *_ = build_dataloader(dataset="cifar10",
                                                  imbalanced_factor=50,
                                                  batch_size=100)

**Loss Function**

The upper-level reweight problem aims to optimize the loss value on the *balanced*
validation dataset (i.e. :code:`reweight_dataloader`) with respect to the *optimal*
parameters of the **Classifier** problem. As before, users can access the inner-level
classifier problem via its name (i.e. :code:`self.classifier`).

.. code:: python

    class Reweight(ImplicitProblem):
        def training_step(self, batch):
            inputs, labels = batch
            outputs = self.classifier(inputs)
            loss = F.cross_entropy(outputs, labels.long())
            print('Reweight Loss:', loss.item())

            return loss

**Training Configuration**

MWN parameters don't affect the loss function of the **Reweight** problem
directly, but only indirectly through the optimal parameters of the classifier
problem. Thus, gradient for MWN should be calculated using hypergradient. In our
example, we use *implicit differentiation with finite difference (a.k.a. DARTS)*
to calculate gradient for MWN parameters. This can be easily specified with
``Config``.

.. code:: python

    reweight_config = Config(type='darts')

**Problem Instantiation**

We can now instantiate the ``Problem`` class for the **Reweight** problem! We use
'reweight' as the ``name`` for this problem.

.. code:: python

    reweight = Reweight(
        name='reweight',
        module=reweight_module,
        optimizer=reweight_optimizer,
        train_data_loader=reweight_dataloader,
        config=reweight_config,
        device="cuda"
    )

|

Engine
------

Recalling the :doc:`Software Design <../../quickstart/concept_software>` chapter,
the ``Engine`` class handles problem dependencies and execution of multilevel
optimization. Let's again take a step-by-step dive into each of these components.

**Problem Dependencies**

The dependency between problems are split into two categories — upper-to-lower (``u2l``)
and lower-to-upper(``l2u``) — both of which are defined using a Python dictionary. In
our example, ``reweight`` is the upper-level problem and ``classifier`` is the
lower-level problem.

.. code:: python

    u2l = {reweight: [classifier]}
    l2u = {classifier: [reweight]}
    dependencies = {'l2u': l2u, 'u2l': u2l}

**Engine Instantiation**

To instantiate the ``Engine`` class, we need to provide all involved problems as well as
the Engine configuration. Since we already defined all problems, we can simply combine
them in a Python list. In addition, we perform our multilevel optimization for 10,000
iterations, which can be specified in ``EngineConfig``.

.. code:: python
    
    from betty.configs import EngineConfig
    from betty.engine import Engine

    problems = [hpo, classifier]
    engine_config = EngineConfig(train_iters=10000)
    engine = Engine(config=engine_config, problems=problems, dependencies=dependencies)

**Execution of Multilevel Optimization**

Finally, multilevel optimization can be excuted by running ``engine.run()``, which calls
the ``step`` method of the lowermost problem (i.e. **Classifier**), which corresponds to a
single step of gradient descent. After unrolling gradient descent for the lower-most
problem for a pre-determined number of steps (``step`` attribute in ``hpo_config``), the
``step`` method of **Classifier** will automatically call the ``step`` method of
**Reweight** according to the provided dependencies.

.. code:: python

    engine.run()

|

Results
-------

The full code of the above example can be found in this
`link <https://github.com/sangkeun00/betty/tree/main/examples/learning_to_reweight>`_.
If everything runs correctly, you should see something like below on your screen:

On long-tailed CIFAR10 image classification benchmark, our MWN implementation achieves:

Table
