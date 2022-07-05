Quick Start
===========

Throughout our tutorials, we will use **Data Reweighting for Long-Tailed Image
Classification** as our running example.  The basic context is that we aim to mitigate
a class imbalance problem (or long-tailed distribution problem) by re-assigning
higher/lower weights to data from rare/common classes. In particular, `Meta-Weight-Net
(MWN) <https://arxiv.org/abs/1902.07379>`_ formulates data reweighting as bilevel
optimization as follows:

.. math::

        w^*=\underset{w}{\mathrm{argmin}}\;\mathcal{L}_{reweight}(\theta^*(w))\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\;\;\,\text{Reweighting}\\
        \text{s.t. }\theta^*(w)=\underset{\theta}{\mathrm{argmin}}\;\frac{1}{N}\sum_{i=1}^n\mathcal{R}(L^i_{class}(\theta);w)\cdot L^i_{class}(\theta)\quad\quad\quad\text{Classification}

where :math:`\theta` denotes the classifier network's parameters, :math:`L_{class}^i` is
the classification loss (cross-entropy) for the :math:`i`-th training sample,
:math:`\mathcal{L}_{reweight}` is the loss for the reweighting level (cross-entropy),
and :math:`w` denotes the parameters for MWN :math:`\mathcal{R}`, which reweights each
training sample given training loss :math:`L^i_{class}`.

Now that we have a problem formulation, we need to (1) define each level problem with
the ``Problem`` class, and (2) define dependencies between problems with the ``Engine``
class.


.. NOTE: the following bar gives a small gap between sections for readability.

|


Basic setup
-----------

Before diving into MLO, we do basic setup such as importing dependencies and
constructing the imbalanced (or long-tailed) dataset. Here, we set the data imbalance
factor to 100, meaning that the most common class has 50 times more data than the least
common class. This part is not directly relevant to MLO, so users can simply copy and
paste the following code.

.. raw:: html

   <details>
   <summary><a>Preparation code</a></summary>

.. code-block:: python

    # import dependencies
    import copy
    import numpy as np

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import MNIST


    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Construct imbalanced (or long-tailed) dataset
    def build_dataset(reweight_size=1000, imbalanced_factor=100):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        dataset = MNIST(root="./data", train=True, download=True, transform=transform)

        num_classes = len(dataset.classes)
        num_meta = int(reweight_size / num_classes)

        index_to_meta = []
        index_to_train = []

        imbalanced_num_list = []
        sample_num = int((len(dataset.targets) - reweight_size) / num_classes)
        for class_index in range(num_classes):
            imbalanced_num = sample_num / (imbalanced_factor ** (class_index / (num_classes - 1)))
            imbalanced_num_list.append(int(imbalanced_num))
        np.random.shuffle(imbalanced_num_list)

        for class_index in range(num_classes):
            index_to_class = [
                index for index, label in enumerate(dataset.targets) if label == class_index
            ]
            np.random.shuffle(index_to_class)
            index_to_meta.extend(index_to_class[:num_meta])
            index_to_class_for_train = index_to_class[num_meta:]

            index_to_class_for_train = index_to_class_for_train[: imbalanced_num_list[class_index]]

            index_to_train.extend(index_to_class_for_train)

        reweight_dataset = copy.deepcopy(dataset)
        dataset.data = dataset.data[index_to_train]
        dataset.targets = list(np.array(dataset.targets)[index_to_train])
        reweight_dataset.data = reweight_dataset.data[index_to_meta]
        reweight_dataset.targets = list(np.array(reweight_dataset.targets)[index_to_meta])

        return dataset, reweight_dataset

    classifier_dataset, reweight_dataset = build_dataset(imbalanced_factor=100)

.. raw:: html

   </details>

|

Problem
-------

In this example, we have a MLO program consisting of two problem levels: *upper* and
*lower*. We respectively refer to these two problems as **Reweight** and **Classifier**,
and create ``Problem`` classes for each of them.  As introduced in the :doc:`Software
Design <../../quickstart/concept_software>` chapter, each problem is defined by (1)
module, (2) optimizer, (3) data loader, (4) loss function, (5) training configuration,
and (6) other optional components (e.g. learning rate scheduler). Everything except for
(4) loss function can be provided through the class constructor, and (4) can be provided
via the ``training_step`` method. In the following subsections, we provide a
step-by-step guide for implementing each of these components in the ``Problem`` class,
for both the lower-level and upper-level problems.

Lower-level Problem (Classifier)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In our data reweighting example, the lower-level problem corresponds to the long-tailed
MNIST image classification task. The data loader code is adopted from `here
<https://github.com/ShiYunyi/Meta-Weight-Net_Code-Optimization/blob/main/noisy_long_tail_CIFAR.py>`_.
We can respectively define the module, optimizer, data loader, loss function, and
training configuration as follows.

**Module, Optimizer, Data Loader, (optional) Scheduler**

.. code:: python

    # Module
    classifier_module = nn.Sequential(
        nn.Flatten(), nn.Linear(784, 200), nn.ReLU(), nn.Linear(200, 10)
    )

    # Optimizer
    classifier_optimizer = optim.SGD(classifier_module.parameters(), lr=0.1, momentum=0.9)

    # Data Loader
    classifier_dataloader = DataLoader(
        classifier_dataset, batch_size=100, shuffle=True, pin_memory=True
    )

    # LR Scheduler
    classifier_scheduler = optim.lr_scheduler.MultiStepLR(
        classifier_optimizer, milestones=[1500, 2500], gamma=0.1
    )

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
**Reweight** problem. The Engine class will provide an access to the **Reweight**
problem via its name for the **Classifier** problem (i.e. in the line :code:`weight =
self.reweight(loss_vector_reshape.detach())`). Thus, users should be aware of names of
other problems, with which the current problem interacts, when writing the loss
function.

**Training Configuration**

The **Reweight** parameter affects optimization of the **Classifier** parameter,
which will again affect the **Reweight** loss function. Thus, best-response Jacobian
for the optimization process of **Classifier** problem should be calculated. In this
tutorial, we adopt *implicit differentiation with finite difference (a.k.a. DARTS)*
as a best-response Jacobian calculation algorithm. Furthermore, since **Classifier**
is the lower-level problem, we need to specify how many steps we want to unroll
before updating the upper-level **Reweight** problem. We choose the simplest
one-step unrolling for our example. All of these can be easily specified with
``Config``.

.. code:: python

    from betty.configs import Config

    classifier_config = Config(type='darts', unroll_steps=1)

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
        device=device
    )

|

Upper-level Problem (Reweight)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While the lower-level problem is a classification problem, the upper-level problem is a
reweighting problem. Specifically, `Meta-Weight-Net (MWN)
<https://arxiv.org/abs/1902.07379>`_ proposes to reweight each data sample using an MLP
with a single hidden layer, which takes a loss value as an input and outputs an
importance weight. 

**Module, Optimizer, Data Loader**

.. code:: python

    # Module
    reweight_module = nn.Sequential(
        nn.Linear(1, 100), nn.ReLU(), nn.Linear(100, 1), nn.Sigmoid()
    )
    
    # Optimizer
    reweight_optimizer = optim.Adam(reweight_module.parameters(), lr=1e-5)

    # Data Loader
    reweight_dataloader = DataLoader(
        reweight_dataset, batch_size=100, shuffle=True, pin_memory=True
    )


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

Since the **Reweight** problem is the uppermost problem, there is no need for
calculating best-response Jacobian. Thus, we don't need to specify any training
configurations for the **Reweight** problem.

.. code:: python

    reweight_config = Config()

**Problem Instantiation**

We can now instantiate the ``Problem`` class for the **Reweight** problem. We use
'reweight' as the ``name`` for this problem.

.. code:: python

    reweight = Reweight(
        name='reweight',
        module=reweight_module,
        optimizer=reweight_optimizer,
        train_data_loader=reweight_dataloader,
        config=reweight_config,
        device=device
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
them in a Python list. In addition, we perform our multilevel optimization for 3,000
iterations, which can be specified in ``EngineConfig``.

.. code:: python
    
    from betty.configs import EngineConfig
    from betty.engine import Engine

    problems = [reweight, classifier]
    engine_config = EngineConfig(train_iters=3000)
    engine = Engine(config=engine_config, problems=problems, dependencies=dependencies)

**Execution of Multilevel Optimization**

Finally, multilevel optimization can be excuted by running ``engine.run()``, which calls
the ``step`` method of the lowermost problem (i.e. **Classifier**), which corresponds to a
single step of gradient descent. After unrolling gradient descent for the lower-most
problem for a pre-determined number of steps (``unroll_steps`` attribute in
``classifier_config``), the ``step`` method of **Classifier** will automatically call
the ``step`` method of **Reweight** according to the provided dependencies.

.. code:: python

    engine.run()

|

Results
-------

Once the training is done, we perform the validation procedure *manually* as below:

.. code:: python

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    valid_dataset = MNIST(root="./data", train=False, transform=transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=100, pin_memory=True)

    correct = 0
    total = 0
    for x, target in valid_dataloader:
        x, target = x.to(device), target.to(device)
        out = classifier(x)
        correct += (out.argmax(dim=1) == target).sum().item()
        total += x.size(0)
    acc = correct / total * 100
    print("Imbalanced Classification Accuracy:", acc)

The full code of the above example can be found in this
`link <https://github.com/sangkeun00/betty/blob/main/tutorial/1_quick_start.py>`_.
If everything runs correctly, you should see something like below on your screen:

.. code:: python

    [2022-06-20 13:01:48] [INFO] Initializing Multilevel Optimization...

    [2022-06-20 13:01:51] [INFO] *** Problem Information ***
    [2022-06-20 13:01:51] [INFO] Name: reweight
    [2022-06-20 13:01:51] [INFO] Uppers: []
    [2022-06-20 13:01:51] [INFO] Lowers: ['classifier']
    [2022-06-20 13:01:51] [INFO] Paths: [['reweight', 'classifier', 'reweight']]

    [2022-06-20 13:01:51] [INFO] *** Problem Information ***
    [2022-06-20 13:01:51] [INFO] Name: classifier
    [2022-06-20 13:01:51] [INFO] Uppers: ['reweight']
    [2022-06-20 13:01:51] [INFO] Lowers: []
    [2022-06-20 13:01:51] [INFO] Paths: []

    [2022-06-20 13:01:51] [INFO] Time spent on initialization: 3.124 (s)

    Classification Accuracy: 95.41

Finally, we compare our data reweighting result with the baseline without reweighting
in the below table:

+---------------+---------------+
|               | Test Accuracy |
+===============+===============+
| Baseline      | 91.82%        |
+---------------+---------------+
| Reweighting   | 95.41%        |
+---------------+---------------+

The above result shows that long-tailed image classification can clearly benefit from
data reweighting!

Happy Multilevel Optimization!
