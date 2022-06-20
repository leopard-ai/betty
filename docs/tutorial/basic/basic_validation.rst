Validation
==========

Unlike traditional neural network training, validation for MLO is defined for the
*whole program* instead of *each level*. Thus, we intentionally design Betty to
handle the validation procedure in the ``Engine`` class, which handles the overall
MLO program. More specifically, the validation procedure can be added by:

1. subclassing the ``Engine`` class
2. implementing a ``validation`` method

In this tutorial, we reuse the data reweighting example from :doc:`basic_start`, and
provide a reference for implementing a validation procedure:

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

        print('acc:', acc, '|| best_acc:', self.best_acc)

        return acc

Users can also specify how often they want to perform a validation procedure via
``EngineConfig``.

.. code:: python

    engine_config = EngineConfig(train_iters=10000, valid_step=100)

Users can simply replace the default ``Engine`` class with their custom
``ReweightEngline`` to instantiate and execute their MLO program.

.. code:: python

    engine = ReweightingEngine(config=engine_config,
                               problems=problems,
                               dependencies=dependencies)
    engine.run()