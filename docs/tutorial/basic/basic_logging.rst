Logging
=======

Monitoring metrics (e.g. training loss) is essentially in the ML development.
Therefore, Betty provides an easy-to-use interface for logging whichever metrics
that users are interested in for both training and validation procedures. In detail,
users can enable logging by returning a Python dictionary in :code:`training_step` of
the ``Problem`` class and :code:`validation` of the ``Engine`` class respectively for
training and validation procedures. In our data reweighting example, this could be
implemented as:

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

        return {"acc": acc, "best_acc": best_acc}

Finally, users can specify how often they want to log these metrics in ``Config``
and ``EngineConfig`` respectively for training (``Problem``) and validation
(``Engine``). If users don't specify the log step in configurations, logging will
be automatically diabled.

.. code:: python

    # Training (Problem)
    reweight_config = Config(type="darts", log_step=100)

    # Validation (Engine)
    engine_config = EngineConfig(train_iters=10000, valid_step=100)

Once logging is enabled, users should be able to see:

.. code::

    asdfa

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