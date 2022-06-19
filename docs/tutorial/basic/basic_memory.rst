Memory Optimization
===================

As MLO involves multiple optimization problems, it tends to use more
GPU memory than traiditional single-level optimization problems. Given the
recent trend of Transformer-based large models (e.g., BERT, GPT), Betty
strives to support large-scale MLO and meta-learning by implementing 
various memory optimization techniques.

Currently, we support three memory optimization techniques:

1. Gradient accumulation
2. FP16 (half-precision) training
3. (non-distributed) Data-parallel training

|

Gradient Accumulation
---------------------

Gradient accumulation is an effetive way to reduce the interemediate states memory, by
accumulating gradients from *multiple small* mini-batches rather than calculating
gradient of *one large* mini-batch. In Betty, gradient accumulation can be enabled for
*each level* problem via ``Config`` as:

.. code:: python

    reweight_config = Config(type="darts", log_step=100, gradient_accumulation=4)

|

FP16 Training
-------------

FP16 (or half-precision) training replaces some of full-precision operations (e.g.
linear) with half-precision operations at the cost of (potential) training instability.
In Betty, users can easily enable FP16 training for *each level* problem via
``Config`` as:

.. code:: python

    reweight_config = Config(type="darts", log_step=100, gradient_accumulation=4, fp16=True)

|

(non-distributed) Data-parallel training
----------------------------------------

Data-parallel training splits large-batch into several small batches across
multiple GPUs and thereby reduce the memory footprint for intermediate states.
While distributed data-parallel training normally achieves much better training
speed, Betty currently only supports non-distributed data-parallel training
via ``EngineConfig``:

.. code:: python

    engine_config = EngineConfig(train_iters=10000, valid_step=100, distributed=True)

|

Memory optimization results
---------------------------
We perform ablation study to analyze how each technique affects GPU memroy usage.
The result is shown in the below table.

Table