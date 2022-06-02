betty.logging package
=====================

Betty provides an automatic logging feature to easily track train loss, validation accuracy, etc.
Users can specify metrics to be logged through ``train_step`` method of ``Problem`` class, and
``validation`` method of ``Engine`` class. Currently, Betty supports two types of loggers:
``TensorBoardLogger`` and ``WandBLogger``.

Tensorboard Logger
------------------

.. automodule:: betty.logging.logger_tensorboard
   :members:
   :undoc-members:

WandB Logger
------------

.. automodule:: betty.logging.logger_wandb
   :members:
   :undoc-members:

Base Logger
---------------------------------

.. automodule:: betty.logging.logger_base
   :members:
   :undoc-members:
