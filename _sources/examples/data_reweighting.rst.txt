Data Reweighting
================
Introduction
------------
Here we re-implement the data reweighting algorithm from
`Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting <https://arxiv.org/abs/1902.07379>`_,
which is a two level MLO program. The first level or lower level
problem is the classification problem and the second level or upper level
problem is the meta learning problem.  These levels will be followed by a
validation stage at the end. 

**Classification Problem:** Here we train the weights :math:`\textbf{w}` of the
classifier by minimizing the loss calculated on the training data set while imposing
some meta weight on each sample loss. Let the :math:`i^{th}` sample loss be
:math:`L_{i}^{train}(\textbf{w}) = l(y_i, y_{predicted})` and the :math:`i^{th}`
meta weight be :math:`\mathcal{V}(L_{i}^{train}(\textbf{w}), \Theta)` then the
objective for this problem will be,

.. math::

    \textbf{w}^{*}(\Theta) = \mathrm{argmin} \mathcal{L}^{train}(\textbf{w} ; \Theta) = \frac{1}{N} \sum_{i=1}^{N}\mathcal{V}(L_{i}^{train}(\textbf{w}), \Theta)L_{i}^{train}(\textbf{w})

The model of the classifier is specified to be ResNet32 and we will be using
the SGD algorithm for the optimization.

**Meta Learning Problem:** Here we train the parameters :math:`\Theta` of the
meta weight net by minimizing the loss calculated on the meta training data set.
Let the :math:`i^{th}` sample loss on the meta data be
:math:`L_{i}^{meta}(\textbf{w}) = l(y^{(meta)}_i, y^{(meta)}_{predicted})` then
the objective of this problem will be,

.. math::
    
    \Theta^{*} = \mathrm{argmin} \mathcal{L}^{meta}(\textbf{w}^{*}(\Theta)) = \frac{1}{M} \sum_{i=1}^{M}L_{i}^{meta}(\textbf{w}^{*}(\Theta))

The model of the meta weight net is chosen to be an MLP and we will be using the Adam
algorithm for the optimization. For complete and detailed formulation of the loss
functions see `here <https://arxiv.org/abs/1902.07379>`__.

Note that for calculating the loss of first level we need the forward pass of the second
level and for calculating the loss of second level we need the forward pass of the first
level. Hence we define the following dependencies. The first level depends on the second
level through a ``'u2l'`` (upper to lower) dependency and the second level depends on the
first level through a ``'l2u'`` (lower to upper) dependency.

Course of Action
----------------
In order to implement the data reweighting algorithm we will go through the following pipeline,

1. **Preparing Data:** Prepare the data that will be used for training.
2. **Designing Models:** Design models that will be used in the two levels.
3. **Using betty:** Finally use Betty to implement the two level MLO program.

Preparing Data
--------------
Here we prepare the data that will be used for training the models in the different
levels of the algorithm. We will require three different data sets. The first is
``train_dataloader`` which will be used in the first level. The second is
``meta_dataloader`` which will be used in the second level. Finally we will have a
``test_dataloader`` which will be used in the validation stage. These data sets can
be prepared as given
`here <https://github.com/sangkeun00/betty/blob/main/examples/learning_to_reweight/data.py>`__.

Designing Models
------------------
Here we design the models used in the levels. We will have to prepare one model
each for our two levels. The first level has the ``ResNet32`` model and the second
level has the ``MLP`` model. Both of these models can be designed as given
`here <https://github.com/sangkeun00/betty/blob/main/examples/learning_to_reweight/model.py>`__.

Using Betty
------------------
Now we will train our models using the data reweighting algorithm with the help
of Betty. We first import the required libraries. The code blocks used
below can be found
`here <https://github.com/sangkeun00/betty/blob/main/examples/learning_to_reweight/main.py>`__.

.. code-block:: python

 import torch
 import torch.nn.functional as F
 import torch.optim as optim

 from model import *
 from data import *
 from utils import *

 from betty.engine import Engine
 from betty.problems import ImplicitProblem
 from betty.configs import Config, EngineConfig

Now we simply need to do two things to implement our algorithm:

1. Define each level's optimization problem using the ``Problem`` class.
2. Define the hierarchical problem structure using the ``Engine`` class.

Defining ``Problem``
^^^^^^^^^^^^^^^^^^^^
Each level problem can be defined with seven components: (1) module, (2) optimizer,
(3) data loader, (4) loss function, (5) problem configuration, (6) name, and
(7) other optional components (e.g. learning rate scheduler). The loss function
(4) can be defined via the ``training_step`` method, while all other components can
be provided through the class constructor.

**First Level:** The first level is characterized by the follwing code. The comments
along with the code assist the understanding.

.. code-block:: python
 
 #all problem classes are supposed to be a subclass of ImplicitProblem
 #the Inner problem class specifies the classifier problem
 class Inner(ImplicitProblem):

    #this method defines the forward pass of the classifier with x as an input
    def forward(self, x):
        #the module attribute of a problem class contains its model
        return self.module(x)

    #this method defines the loss function of our problem
    #it takes a batch (subset) of (inputs, labels) from the training data set of the problem as input
    def training_step(self, batch):
        inputs, labels = batch

        #we calculate the predicted labels from the forward pass of the classifier
        outputs = self.forward(inputs)

        #we calculate the cross entropy loss of our classifier probelem and reshape it as required
        loss_vector = F.cross_entropy(outputs, labels.long(), reduction="none")
        loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))

        #we calculate the weight that is supposed to be imposed on every sample loss
        #we do so by using the forward pass of the second level problem
        #we can access the forward pass of other problems by using the 'name' attribute
        weight = self.outer(loss_vector_reshape.detach())

        #we calculte the final loss as the mean of the product of the weights and indvidual
        #sample losses
        loss = torch.mean(weight * loss_vector_reshape)

        return loss

    #this method sets the training data of the problem
    def configure_train_data_loader(self):
        return train_dataloader

    #this method sets the module of the problem to the required model
    def configure_module(self):
        return ResNet32(args.dataset == "cifar10" and 10 or 100).to(device=args.device)

    #this method sets the optimizer of the problem
    #we have used the SGD algorithm for optimization here
    def configure_optimizer(self):
        optimizer = optim.SGD(
            self.module.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            dampening=args.dampening,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
        return optimizer

    #this method sets the scheduler sepecifications of the problem (optional)
    def configure_scheduler(self):
        scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[5000, 7500, 9000], gamma=0.1
        )
        return scheduler

**Second Level:** The first level is characterized by the follwing code. The comments
along with the code assist the understanding.

.. code-block:: python

 #all problem classes are supposed to be a subclass of ImplicitProblem
 #the Outer problem class specifies the meta learning problem
 class Outer(ImplicitProblem):

    #this method defines the forward pass of the meta learning problem with x as an input
    def forward(self, x):
        #the module attribute of a problem class contains its model
        return self.module(x)

    #this method defines the loss function of our problem
    #it takes a batch (subset) of (inputs, labels) from the meta data set of the problem as input
    def training_step(self, batch):
        inputs, labels = batch

        #we calculate the predicted labels from the forward pass of the classifier
        #we do so by using the forward pass of the second level problem
        #we can access the forward pass of other problems by using the 'name' attribute
        outputs = self.inner(inputs)

        #we calculte the final loss as the mean of the product of the weights and
        #indvidual sample losses
        loss = F.cross_entropy(outputs, labels.long())

        #we calculate the accuracy of the predictions made
        acc = (outputs.argmax(dim=1) == labels.long()).float().mean().item() * 100

        #we return the loss and the accuracy in form of a dictionary
        return {"loss": loss, "acc": acc}

    #this method sets the training data of the problem
    def configure_train_data_loader(self):
        return meta_dataloader

    #this method sets the module of the problem to the required model
    def configure_module(self):
        meta_net = MLP(
            hidden_size=args.meta_net_hidden_size, num_layers=args.meta_net_num_layers
        ).to(device=args.device)
        return meta_net

    #this method sets the optimizer of the problem
    #we have used the Adam algorithm for optimization here
    def configure_optimizer(self):
        meta_optimizer = optim.Adam(
            self.module.parameters(), lr=args.meta_lr, weight_decay=args.meta_weight_decay
        )
        return meta_optimizer

**Instantiation:** here we instantiate our porblem classes and make their respective
objects which call their constructors.

.. code-block:: python

    #we difine the configurations of both the problems using the Config library
    #configuration of a prooblem contains important specifications related to the problem
    outer_config = Config(type="darts", fp16=args.fp16, log_step=100)
    inner_config = Config(type="darts", fp16=args.fp16, unroll_steps=1)

    #we instantiate the Inner and Outer problems and set their 'name', 'config',
    #'device' attributes
    outer = Outer(name="outer", config=outer_config, device=args.device)
    inner = Inner(name="inner", config=inner_config, device=args.device)

With this our problems are characterized and instansiated. Now we move on to set
our ``Engine`` class.

Defining ``Engine``
^^^^^^^^^^^^^^^^^^^

The Engine class handles the hierarchical dependencies between problems. In MLO, there
are two types of dependencies: upper-to-lower ``'u2l'`` and lower-to-upper ``'l2u'``.
Both types of dependencies can be defined with Python dictionary, where the key is the
starting node and the value is the list of destination nodes.

Since Engine manages the whole MLO program, you can also perform a global validation stage
within it. All involved problems of the MLO program can again be accessed with their
'name' attribute.

.. code-block:: python

    #initiate best accuracy
    best_acc = -1

    #when we have to define a validation level then we make a subclass of Engine to do so
    #if a validation level is not required we do not need this class
    class ReweightingEngine(Engine):
        @torch.no_grad()

        #defines the validation level
        def validation(self):

            #initiate correct number of predictions and total predictions
            correct = 0
            total = 0
            global best_acc

            #go thorugh the testing data set for validation
            for x, target in test_dataloader:

                #move the inputs and labels to the desired device
                x, target = x.to(args.device), target.to(args.device)

                #calculate the predicted labels without gradient tracking
                with torch.no_grad():
                    out = self.inner(x)
                
                #update correct if the prediction is correct
                correct += (out.argmax(dim=1) == target).sum().item()

                #update total
                total += x.size(0)
            
            #calculate accuracy
            acc = correct / total * 100

            #update best accuracy if the new accuracy is greater than the previous accuracy
            if best_acc < acc:
                best_acc = acc

            #return accuracy and best accuracy as a dictionary
            return {"acc": acc, "best_acc": best_acc}

    #setup engine configuration using EngineConfig Library
    engine_config = EngineConfig(train_iters=10000, valid_step=100, distributed=args.distributed, roll_back=args.rollback)

    #specify all the problems in a list
    problems = [outer, inner]

    #set dependencies as dictionaries
    #level 1(inner) accesses level 2(outer) 
    u2l = {outer: [inner]}

    #level 2(outer) accesses level 1(inner)
    l2u = {inner: [outer]}

    #set up a dictiontionary to list out dependencies
    dependencies = {"l2u": l2u, "u2l": u2l}

    #instantiate engine and set the 'config', 'problems', 'dependencies' attributes
    engine = ReweightingEngine(config=engine_config, problems=problems, dependencies=dependencies)

    #run the engine
    engine.run()
    print(f"IF {args.imbalanced_factor} || Best Acc.: {best_acc}")

With this the dependencies are defined and ``.run()`` method of ``Eninge`` class
will start the program.

Conclusion
----------

Once we define all optimization problems and the hierarchical dependencies between the
problems with, respectively, the Problem class and the Engine class, all complicated
internal mechanism of MLO such as gradient calculation and optimization execution order
are handled by Betty.
