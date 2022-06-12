Software Design
===============

Betty allows for an easy-to-use, modular, and maintainable programming interface for
multilevel optimization (MLO) by breaking down MLO into two high-level concepts --- (1)
optimization problems, and (2) problem dependencies --- for which we design two abstract
Python classes:

- ``Problem`` class: an abstraction of optimization problems.
- ``Engine`` class: an abstraction of problem dependencies.

In this chapter, we will introduce each of these concepts/classes in depth.

Problem
-------

Under our abstraction, each optimization problem :math:`P` in MLO is defined by the (1)
module, (2) the optimizer, (3) the data loader, (4) the sets of the upper and lower
constraining problems, (5) the loss function, (6) the problem (or optimization)
configuration, (7) the name, and (8) other optional components.  The example usage of
the ``Problem`` class is shown below:

.. code:: python

    """ Setup of module, optimizer, and data loader """
    my_module, my_optimizer, my_data_loader = problem_setup()

    class MyProblem(ImplicitProblem):
        def training_step(self, batch):
            """ Users define the loss function here """
            loss = loss_fn(batch, self.module, self.other_probs, ...)
            acc = get_accuracy(batch, self.module, ...)
            return {'loss': loss, 'acc': acc}
        
    """ Optimization Configuration """
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
~~~~~~~~~~~~~~~~
Automatic differentiation for multilevel optimization can be roughly categorized into
two types: iterative differentiation (ITD) and implicit differentiation (AID). While AID
allows users to use native PyTorch modules and optimizers, ITD requires patching both
modules and optimizers to follow a functional programming paradigm. Due to this
difference, we provide separate classes ``IterativeProblem`` and ``ImplicitProblem``
respecitvely for ITD and AID. Empirically, we observe that AID often achieves better
memory efficiency, training wall time, and final accuracy. Thus, we highly recommend
using the ``ImplicitProblem`` class as a default setting.

(1) Module
~~~~~~~~~~
The module defines the parameters to be learned in the current optimization problem, and
corresponds to :math:`\theta_k` in our mathematical formulation (:doc:`Chapter
<concept_mlo>`). In practice, the module is usually defined using PyTorch's
``torch.nn.Module``, and is passed to the ``Problem`` class through the constructor.

(2) Optimizer
~~~~~~~~~~~~~
The optimizer updates parameters for the above module. In practice, the optimizer is
most commonly defined using PyTorch's ``torch.optim.Optimizer``, and is also passed to
the ``Problem`` class through the constructor.

(3) Data loader
~~~~~~~~~~~~~~~
The data loader defines the associated training data, denoted :math:`\mathcal{D}_k` in
our mathematical formulation. It is normally defined using PyTorch's
``torch.utils.data.DataLoader``, but it can be any Python ``Iterator``. The data loader
can also be provided through the class constructor.

(4) Upper & Lower Constraining Problem Sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
While the upper & lower constraining problem sets
:math:`\mathcal{U}_k\;\&\;\mathcal{L}_k` are at the core of our mathematical
formulation, we don't allow users to directly specifiy them in the ``Problem`` class.
Rather, we design Betty so that the constraining sets are provided directly from
``Engine``, the class where all problem dependencies are handled. In doing so, users
need to provide the hierarchical problem dependencies only once when they initialize
``Engine``, and can avoid the potentially error-prone and cumbersome process of
provisioning constraining problems manually every time they define new problems.

(5) Loss function
~~~~~~~~~~~~~~~~~
The loss function defines the optimization objective :math:`\mathcal{C}_k` in our
formulation.  Unlike previous components, the loss function is defined through the
``training_step`` method as shown above. In addition, the ``training_step`` method
provides an option to define other metrics (e.g.  accuracy in image classification),
which can be returned with the Python dictionary. When the return type is not a Python
dictionary, the API will assume that the returned value is the loss by default.
Furthermore, the returned dictionary/value of ``training_step`` will be automatically
logged with our logger to a visualization tool (e.g. tensorboard) as well as the
standard output stream (i.e. print in the terminal). Our ``training_step`` method is
highly inspired by `PyTorch Lightning's
<https://github.com/PyTorchLightning/pytorch-lightning>`_ ``training_step`` method.

(6) Optimization Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unlike automatic differentiation in neural networks, autodiff in MLO requires
approximating gradients with, for example, implicit differentiation. Since there can be
different approximation methods and configurations, we allow users to specify all
choices through the ``Config`` data class. In addition, ``Config`` allows users to
specify other training details such as gradient accumulation steps, logging steps, and
fp16 training options.  We provide the default value for each attribute in ``Config``,
so, in most cases, users will only need to specify 3-4 attributes based on their needs.

(7) Name
~~~~~~~~
Users oftentimes need to access constraining problems
:math:`\mathcal{U}_k\;\&\;\mathcal{L}_k` when defining the loss function in
``training_step``. However, since constraining problems are directly provided by the
``Engine`` class, users lack the way to access constraining problems from the current
problem. Thus, we design the ``name`` attribute, through which users can access other
problems in the ``Problem`` and the ``Engine`` classes. For example, when your MLO
involves ``Problem1(name='prob1', ...)`` and ``Problem2(name='prob2', ...)``, you can
access ``Problem2`` from ``Problem1`` with ``self.prob2``.

(8) Other Optional Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
While not considered essential components, learning rate schedulers or parameter
callbacks (e.g. parameter clipping/clamping) can optionally be provided by users as
well. Interested users can refer to the API documentation for these features.

Engine
------

While ``Problem`` manages each optimization problem, ``Engine`` handles a dataflow graph
based on the user-provided hierarchical problem dependencies. An example usage of the
``Engine`` class is provided below:

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

Here, we take a deeper look into each component of ``Engine``.

(1) Problems
~~~~~~~~~~~~
Users should provide all of the involved optimization problems through the `problems`
argument.

(2) Dependencies
~~~~~~~~~~~~~~~~
As discussed in :doc:`this section <concept_mlo>`, MLO has two types of dependencies
between problems: upper-to-lower and lower-to-upper. We allow users to define two
separate graphs, one for each type of edge, using a Python dictionary, in which
keys/values respectively represent start/end nodes of the edge. When user-defined
dependency graphs are provided, ``Engine`` compiles them and finds all paths required
for automatic differentiation with a modified depth-first search algorithm.  Moreover,
``Engine`` determines constraining problem sets for each problem based on the dependency
graphs, as mentioned above.

(3) Validation
~~~~~~~~~~~~~~
We currently allow users to define one validation stage for the *whole* multilevel
optimization program. This can be achieved by implementing the ``validation`` method in
``Engine`` as shown above. As in the ``training_step`` method of the ``Problem`` class,
users can return whichever metrics they want to log with the Python dictionary.

(4) Engine Configuration
~~~~~~~~~~~~~~~~~~~~~~~~
Users can specify several configurations for the whole multilevel optimization program,
such as the total training iterations, the validation step, and the logger type.

(5) Run
~~~~~~~
Once all initialization steps are complete, users can run the MLO program by calling the
Engine's ``run`` method, which repeatedly calls ``step`` methods of lowermost problems.
The ``step`` methods of upper-level problems will be automatically called from the
``step`` methods of lower-level problems following lower-to-upper edges.


To summarize, Betty provides a PyTorch-like programming interface for defining multiple
optimization problems, which can scale up to large MLO programs with complex
dependencies, as well as a modular interface for a variety of best-response Jacobian
algorithms, without requiring mathematical and programming proficiency.
