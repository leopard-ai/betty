Logging
=======

Monitoring metrics (e.g. training loss) is essential in ML development.  Betty provides
an easy-to-use interface for flexibly logging whichever metrics are of interest to
users, for both training and validation procedures. Specifically, users can enable
logging by returning a Python dictionary in :code:`training_step` of the ``Problem``
class and :code:`validation` of the ``Engine`` class, respectively, for training and
validation procedures. In our data reweighting example from :doc:`basic_start`, this can
be implemented as:

**Training (Problem)**

.. code:: python

    class Reweight(ImplicitProblem):
        def training_step(self, batch):
            inputs, labels = batch
            outputs = self.classifier(inputs)
            loss = F.cross_entropy(outputs, labels.long())
            acc = (outputs.argmax(dim=1) == labels.long()).float().mean().item() * 100

            return {"loss": loss, "acc": acc}

**Validation (Engine)**

.. code:: python

     class ReweightingEngine(Engine):
        @torch.no_grad()
        def validation(self):
            correct = 0
            total = 0
            if not hasattr(self, best_acc):
                self.best_acc = -1
            for x, target in test_dataloader:
                out = self.inner(x)
                correct += (out.argmax(dim=1) == target).sum().item()
                total += x.size(0)
            acc = correct / total * 100
            if self.best_acc < acc:
                self.best_acc = acc

            return {"acc": acc, "best_acc": self.best_acc}

Finally, users can specify how often they want to log these metrics in ``Config`` and
``EngineConfig``, respectively, for training (``Problem``) and validation (``Engine``).
If users don't specify the log step in configurations, logging will be automatically
disabled.

.. code:: python

    # Training (Problem)
    reweight_config = Config(type="darts", log_step=100)

    # Validation (Engine)
    engine_config = EngineConfig(train_iters=3000, valid_step=100)

When logging is enabled, users should see the following output upon executing
``engine.run()``:

.. code::

    [2022-06-20 13:52:21] [INFO] Initializing Multilevel Optimization...

    [2022-06-20 13:52:24] [INFO] *** Problem Information ***
    [2022-06-20 13:52:24] [INFO] Name: reweight
    [2022-06-20 13:52:24] [INFO] Uppers: []
    [2022-06-20 13:52:24] [INFO] Lowers: ['classifier']
    [2022-06-20 13:52:24] [INFO] Paths: [['reweight', 'classifier', 'reweight']]

    [2022-06-20 13:52:24] [INFO] *** Problem Information ***
    [2022-06-20 13:52:24] [INFO] Name: classifier
    [2022-06-20 13:52:24] [INFO] Uppers: ['reweight']
    [2022-06-20 13:52:24] [INFO] Lowers: []
    [2022-06-20 13:52:24] [INFO] Paths: []

    [2022-06-20 13:52:24] [INFO] Time spent on initialization: 3.126 (s)

    [2022-06-20 13:52:27] [INFO] [Problem "reweight"] [Global Step 100] [Local Step 100] loss: 0.9833700656890869 || acc: 72.99999594688416
    [2022-06-20 13:52:27] [INFO] [Validation] [Global Step 100] acc: 72.39999999999999 || best_acc: 72.39999999999999
    [2022-06-20 13:52:30] [INFO] [Problem "reweight"] [Global Step 200] [Local Step 200] loss: 0.5147801637649536 || acc: 88.99999856948853
    [2022-06-20 13:52:30] [INFO] [Validation] [Global Step 200] acc: 85.22 || best_acc: 85.22
    [2022-06-20 13:52:32] [INFO] [Problem "reweight"] [Global Step 300] [Local Step 300] loss: 0.4090099036693573 || acc: 87.99999952316284
    [2022-06-20 13:52:33] [INFO] [Validation] [Global Step 300] acc: 89.31 || best_acc: 89.31
    [2022-06-20 13:52:35] [INFO] [Problem "reweight"] [Global Step 400] [Local Step 400] loss: 0.6072959899902344 || acc: 89.99999761581421
    [2022-06-20 13:52:36] [INFO] [Validation] [Global Step 400] acc: 90.88000000000001 || best_acc: 90.88000000000001
    [2022-06-20 13:52:38] [INFO] [Problem "reweight"] [Global Step 500] [Local Step 500] loss: 0.32159245014190674 || acc: 93.00000071525574
    [2022-06-20 13:52:39] [INFO] [Validation] [Global Step 500] acc: 90.41 || best_acc: 90.88000000000001
    ...

|

Visualization
-------------

In addition to logging in the terminal, we allow users to visualize metrics
with visualization tools such as
`TensorBoard <https://pytorch.org/docs/stable/tensorboard.html>`_. Users only
need to specify the type of the visualization tool they want in ``EngineConfig``
as:

.. code:: python

    EngineConfig(train_iters=10000, valid_step=100, logger_type='tensorboard')

For example, with the ``tensorboard`` option, visualization results will be saved
in ``./betty_tensorboard`` and can be opened with

.. code::

    tensorboard --logdir=betty_tensorboard

Currently, we only support
`TensorBoard <https://pytorch.org/docs/stable/tensorboard.html>`_ and
`Weights & Biases <https://github.com/wandb/client>`_ for visualizaiton tools.
