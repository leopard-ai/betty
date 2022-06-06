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
As introduced in the `Software Design <concept_software>` chapter, each problem is defined by (1)
module, (2) optimizer, (3) data loader, (4) loss function, (5) training configuration, and (6)
other optional components (e.g. learning rate scheduler). Everything except for (4) loss function
can be provided through the class constructor, and (4) can be provided via the ``training_step``
method. In the following subsections, we provide a step-by-step guide for identifying and
implementing each of these components in the ``Problem`` class.
step-by-step guide for implementing the ``Problem``

Lower-level Problem (Classifier)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In this example, the lower-level problem corresponds to the MNIST image classification task. Thus,
(1) module, (2) optimizer, (3) data loader, (4) loss function, (5) training configuration can be
respectively defined as below.

**(1) Module**

We use the simple MLP with one hidden layer as our classification network (i.e. 784-200-100). This
network can be implemented as:

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

**(2) Optimizer**

We use the SGD optimizer with the learning rate of 0.01 and the momentum value of 0.9 as our
optimizer.

.. code:: python

    classifier_optimizer = optim.SGD(
        classifier_module.parameters(),
        lr=0.01,
        momentum=0.9
    )

**(3) Data Loader**
MNIST dataset and data loader can be easily implemented with ``torchvision`` and
``torch.utils.DataLoader``.

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

**(4) Loss Function**
Unlike other components, loss function should be directly implemented in the ``Problem`` class via
the ``training_step`` method. In our example, loss function is composed of two parts: cross-entropy
classification loss and L2 regularization loss. We also define the ``forward`` method to define the
``__call__`` method of the class.

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

**(5) Training Configuration**
Since the classification problem is the lowest-level problem, it doesn't require any best-response
Jacobian calculation from the lower-level problems. Rather, it would use the PyTorch's default
autograd to calculate the gradient. Therefore, we don't need to specify anything for the
training configuration for this problem.

.. code:: python

    from betty.configs import Config

    classifier_config = Config()

**(6) Problem Instatntiation**
Now that we have all the components to define the problem, we can instantiate the ``Problem`` class.
We use 'classifier' as the ``name`` for this problem.

.. code:: python

    classifier = Classifier(
        name='classifier',
        module=classifier_module,
        optimizer=classifier_optimizer,
        train_data_loader=classifier_data_loader,
        config=classifier_config,
        device="cuda"
    )

Upper-level Problem (HPO)
~~~~~~~~~~~~~~~~~~~~~~~~~
We can repeat the same process with the lower-level problem for the upper-level problem
(HPO).

.. code:: python

    """ (1) module """
    class WeightDecay(nn.Module):
        def __init__(self):
            super(WeightDecay, self).__init__()
            self.fc1_wdecay = nn.Parameter(torch.ones(200, 784) * 5e-4)
            self.fc2_wdecay = nn.Parameter(torch.ones(10, 200) * 5e-4)

        def forward(self):
            return self.fc1_wdecay, self.fc2_wdecay
    hpo_module = WeightDecay()

    """ (2) optimizer """
    hpo_optimizer = optim.Adam(hpo_module.parameters(), lr=1e-5)

    """ (3) data loader """
    hpo_data_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=64,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    """ (4) loss function """
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

            return loss

        def param_callback(self, params):
            # ensure weight decay value >= 0
            for p in params:
                p.data.clamp_(min=1e-8)

    """ (5) training configurations """
    hpo_config = Config(type='darts', step=1, first_order=True, retain_graph=True)

    """ Problem Instantiation """
    hpo = HPO(
        name='hpo',
        module=hpo_module,
        optimizer=hpo_optimizer,
        train_data_loader=hpo_data_loader,
        config=hpo_config,
        device="cuda"
    )

For the ``HPO`` class, we additionally define ``param_callback`` method to ensure that the weight
decay value is always positive by clamping its value.


Engine
------
Now that we defined both level optimization problems with ``Problem``, we inject the dependency
between these problems and optionally the validation stage via the ``Engine`` class. Specifically,
the dependency between problems are split into two categories of upper-to-lower (``u2l``) and
lower-to-upper(``l2u``), and both are defined with the Python dictionary. Finally, the whole
multilevel optimization procedure can be excuted by the ``run`` method of ``Engine``.

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

    problems = [classifier, hpo]
    
    u2l = {hpo: [classifier]}
    l2u = {classifier: [hpo]}
    dependencies = {'l2u': l2u, 'u2l': u2l}

    engine_config = EngineConfig(train_iters=5000, valid_step=100)
    engine = HPOEngine(config=engine_config, problems=problems, dependencies=dependencies)
    engine.run()


Results
-------
We finally compare the test accuracy of our HPO framework with the test accuracy of the baseline
experiment which uses a single weight decay value of :math:`5e^{-4}` in the below table.

Table

The full code of the above example can be found `here <.>`_.