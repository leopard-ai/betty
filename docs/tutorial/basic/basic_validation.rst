Validation
==========

Unlike traditional neural network training, validation for MLO is defined for the *whole
program* instead of *each level*. Thus, we intentionally design Betty to handle the
validation procedure in the ``Engine`` class, which handles the overall MLO program.
More specifically, the validation procedure can be implemented by:

1. Subclassing the ``Engine`` class.
2. Implementing a ``validation`` method.

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

        print('acc:', acc, 'best_acc:', self.best_acc)

Users can also specify how often they want to perform a validation procedure via
``EngineConfig``.

.. code:: python

    engine_config = EngineConfig(train_iters=3000, valid_step=100)

Users can simply replace the default ``Engine`` class with their custom
``ReweightingEngine`` to instantiate and execute their MLO program.

.. code:: python

    engine = ReweightingEngine(config=engine_config,
                               problems=problems,
                               dependencies=dependencies)
    engine.run()

If implemented correctly, users should expect to see:

.. code::

    [2022-06-20 13:29:08] [INFO] Initializing Multilevel Optimization...

    [2022-06-20 13:29:11] [INFO] *** Problem Information ***
    [2022-06-20 13:29:11] [INFO] Name: reweight
    [2022-06-20 13:29:11] [INFO] Uppers: []
    [2022-06-20 13:29:11] [INFO] Lowers: ['classifier']
    [2022-06-20 13:29:11] [INFO] Paths: [['reweight', 'classifier', 'reweight']]

    [2022-06-20 13:29:11] [INFO] *** Problem Information ***
    [2022-06-20 13:29:11] [INFO] Name: classifier
    [2022-06-20 13:29:11] [INFO] Uppers: ['reweight']
    [2022-06-20 13:29:11] [INFO] Lowers: []
    [2022-06-20 13:29:11] [INFO] Paths: []

    [2022-06-20 13:29:11] [INFO] Time spent on initialization: 3.099 (s)

    acc: 81.25 best_acc: 81.25
    [2022-06-20 13:29:14] [INFO] [Validation] [Global Step 100]
    acc: 82.44 best_acc: 82.44
    [2022-06-20 13:29:17] [INFO] [Validation] [Global Step 200]
    acc: 85.53 best_acc: 85.53
    [2022-06-20 13:29:20] [INFO] [Validation] [Global Step 300]
    acc: 88.67 best_acc: 88.67
    [2022-06-20 13:29:23] [INFO] [Validation] [Global Step 400]
    acc: 91.64 best_acc: 91.64
    [2022-06-20 13:29:26] [INFO] [Validation] [Global Step 500]
    acc: 88.44 best_acc: 91.64
    ...
