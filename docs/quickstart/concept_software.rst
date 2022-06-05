Software Design
===============

Betty allows an easy-to-use, modular, and maintainable programming interface for multilevel
optimization (MLO) by breaking down MLO into two high-level concepts --- (1) Optimization Problems,
and (2) Problem Dependencies --- for which we design abstract Python classes:

- ``Problem`` class: Abstraction of optimization problems
- ``Engine`` class: Abstraction of problem dependencies

In this chapter, we will introduce each of these concepts/classes in depth.

Problem
-------

Under our abstraction, each optimization problem :math:`P` in MLO is defined by the (1)
module :math:`\theta_k`, (2) the optimizer, (3) the data loader :math:`D_k`,
(4) the loss function :math:`\mathcal{C}_k`, (5) the problem (or optimization) configuration,
and the sets of the upper and lower constraining problems :math:`\mathcal{U}_k\;\&\;\mathcal{L}_k`.
The example usage of the ``Problem`` class is shown below:

.. code:: python

    """ module, opitmizer, data loader setup """
    my_module, my_optimizer, my_data_loader = problem_setup()

    class MyProblem(ImplicitProblem):
        def training_step(self, batch):
            """ Users define the loss function here """
            loss = loss_fn(batch, self.module, self.other_probs, ...)
            acc = get_accuracy(batch, self.module, ...)
            return {'loss': loss, 'acc': acc}
        
    """ Optimizaiton Configuration """
    config = Config(type="darts", steps=5, first_order=True, retain_graph=True)

    """ Problem Instantiation """
    prob = MyProblem(
        name='myproblem',
        module=my_module,
        optimizer=my_optimizer,
        train_data_loader=my_data_loader,
        config=config,
        device=device
    )

To better understand the ``Problem`` class, we take a deeper dive into each component.

(0) Problem type
Automatic differentiation for multilevel optimization can be roughly categorized into two types:
iterative differentiation (ITD) and implicit differentiation (AID). While AID allows users to use
native PyTorch modules and optimizers, ITD requires patching both modules and optimizers to follow
a functional programming paradigm. Due to this difference, we provide separate classes
``IterativeProblem`` and ``ImplicitProblem`` respecitvely for AID and ITD. Empirically, we observe
that AID achieves better memory efficieny, training wall time, and final accuracy. Thus, we highly
recommend using ``ImplicitProblem`` class as a default setting.

(1) Module
~~~~~~~~~~
Module defines parameters to be learned at the current level :math:`k`, and corresponds to
:math:`\theta_k` in our mathematical formulation (:doc:`Chapter <concept_mlo>`). In practice,
module is defined with PyTorch's ``torch.nn.Module`` as in traditional neural network
implementations, and is passed to the class through the constructor.

(2) Optimizer
~~~~~~~~~~~~~
Optimizer updates parameters for the above module. In practice, module is most commonly defined
with PyTorch's ``torch.optim.Optimizer``, and is also passed to the class through the constructor.

(3) Data loader
~~~~~~~~~~~~~~~
Data loader defines the associated training data :math:`\mathcal{D}_k` in our mathematical
formulation. It is normally defined with PyTorch's ``torch.utils.data.DataLoader``, but it can be
any Python ``Iterators``. Data loader can also be provided through the class constructor.

(4) Upper & Lower Constraining Problem Sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
While upper & lower constraining problem sets :math:`\mathcal{U}_k\;\&\;\mathcal{L}_k` are at the
core of our mathematical formulation, we don't allow users to directly specifiy them in the
``Problem`` class. Rather, we design Betty that the constraining sets are provided directly from
``Engine``, the class where all problem dependencies are handled. In doing so, users need to
provide the hierarchical problem dependencies only once when they initialize ``Engine``, and can
avoid the potentially error-prone and cumbersome process of provisioning constraining problems
manually every time they define new problems.

(5) Loss function
~~~~~~~~~~~~~~~~~
Loss function defines the optimization objective :math:`\mathcal{C}_k` in our formulation.
Unlike previous components, loss function is defined through the ``training_step`` method as shown
above. In addition, ``training_step`` method provides an option to define other metrics such as
accuracy in image classification, which can be returned with the Python dictionary. When the return
type is not Python dictionary, the API will assume that the returned value is loss by default.
Furthremore, returned dictionary/value of ``training_step`` will be automatically logged with our
logger to the visualization tool (e.g. tensorboard) as well as standard output stream (i.e. print
in the terminal). Our ``training_step`` method is highly inspired by
`PyTorch Lightning's <https://github.com/PyTorchLightning/pytorch-lightning>`_
``training_step`` method.

(6) Optimization Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unlike automatic differentiation in neural networks, autodiff in MLO requires approximating
gradient with, for example, implicit differentiation. Since there can be different approximation
methods and configurations, we allow users to specify all these required information through the
``Config`` data class. In addition, ``Config`` allows users to specify other training details such
as gradient accumulation step, logging step, and fp16 training options.
We provide the default value for each attribute in ``Config``, so, in most cases, users only need
to specify 3-4 attributes based on their needs.

(7) Name
~~~~~~~~
Users oftentimes need to access constraining problems :math:`\mathcal{U}_k\;\&\;\mathcal{L}_k` when
defining loss function in ``training_step``. However, since constraining problems are directly
provided by the ``Engine`` class, users lack the way to access constraining problems from the
current problem. Thus, we design the ``name`` attribute, through which users can access other
problems in the ``Problem`` and the ``Engine`` classes. For example, when your MLO involves
``Problem1(name='prob1', ...)`` and ``Problem2(name='prob2', ...)``, you can access
``Problem2`` from ``Problem1`` with ``self.prob2``.

(8) Miscellaneous
~~~~~~~~~~~~~~~~~
While not considered as essential components, learning rate scheduler or parameter callback
(e.g. parameter clipping/clamping) can optionally be provided by users as well. Interested users can
refer to the API documentation.

Engine
------

While ``Problem`` manages each optimization problem, ``Engine`` handles a dataflow graph based on
the user-provided hierarchical problem dependencies. The example usage of the ``Engine`` class is
provided below:

.. code:: python

    class MyEngine(Engine):
        @torch.no_grad()
        def validation(self):
            val_loss = loss_fn(self.prob1, self.prob2, test_loader)
            val_acc = acc_fn(self.prob1, self.prob2, test_loader)

            return {'loss': val_loss, 'acc': val_acc}

    p1 = Problem1(name='prob1', ...)
    p2 = Problem2(name='prob2', ...)
    dependencies = {"u2l": {p1: [p2]}, "l2u": {p1: [p2]}}
    engine_config = EngineConfig(train_iters=5000, valid_step=100)
    engine = MyEngine(problems=[p1, p2], dependencies=dependencies, config=engine_config)
    engine.run()

We here also take a deeper look into each component of ``Engine``.

(1) Problems
~~~~~~~~~~~~
Users should provide all the involved optimization problems through the problem argument.

(2) Dependencies
~~~~~~~~~~~~~~~~
As discussed in :doc:`this Chapter <concept_mlo>`, MLO has two types of dependencies between
problems: upper-to-lower and lower-to-upper. We allow users to define two separate graphs, one for
each type of edge, using a Python dictionary, in which keys/values respectively represent start/end
nodes of the edge. When user-defined dependency graphs are provided, ``Engine`` compiles them and
finds all paths required for automatic differentiation with a modified depth-first search algorithm.
Moreover, ``Engine`` sets constraining problem sets for each problem based on the dependency graphs,
as mentioned above.

(3) Validation
~~~~~~~~~~~~~~
We currently allow users to define one validation stage for the *whole* multilevel optimization
program. This can be achieved by implementing the ``validation`` method in ``Engine`` as shown
above. As in the ``training_step`` method of the ``Problem`` class, users can return whichever
metrics they want to log with the Python dictionary.

(4) Engine Configuration
~~~~~~~~~~~~~~~~~~~~~~~~
Users can specify several configurations for the whole multilevel optimization program, such as
the total training iterations, the validation step, and the logger type.

(5) Run
~~~~~~~
Once all initialization processes are done, users can run a whole MLO program by calling the
``run`` method, which repeatedly calls ``step`` methods of lowermost problems. The ``step`` methods
of upper-level problems will be automatically called from the ``step`` methods of lower-level
problems following lower-to-upper edges.


To summarize, Betty provides a PyTorch-like programming interface of defining multiple optimization
problems, which can scale up to large MLO programs with complex dependencies, as well as a modular
interface for a variety of best-response Jacobian algorithms, without requiring mathematical and
programming proficiency.
