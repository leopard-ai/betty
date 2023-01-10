Distributed Training
====================

Despite the popularity of meta-learning, research in meta-learning has largely been limited to
small-scale setups due to the memory and computation bottlenecks. In detail, meta-learning
oftentimes requires second-order gradient information (i.e., Hessian) and/or multiple
forward/backward computations.

Distributed training emerges as a natural solution to mitigate above issues. To keep up with the
recent trend of large foundation models, Betty supports various distributed training strategies
for meta-learning such as distributed data parallel (DDP) and ZeRO redundancy optimizer (ZeRO).
Most importantly, users can enable these featuers with (1) *one-liner change* in
``EngineConfig``, and (2) launch distributed training PyTorch's native elastic launcher like
``torchrun``.


EngineConfig
------------

In Betty, we provide mulitple distributed training strategies. In v0.2.0, Betty *stably*
supports:

- Distributed Data Parallel (`DDP <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_).
- Zero Redundancy Optimizer (`ZeRO <https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html>`_).

and also experimentally supports:

- Fully Sharded Data Parallel (`FSDP <https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html>`_).

To enable above features, users can simply need to set ``strategy`` attribute in ``EngineConfig``
to user's preferred distributed training strategy as follows.

.. code:: python

    # DDP
    engine_config = EngineConfig(strategy="distributed", train_iters=3000, ...)

    # ZeRO
    engine_config = EngineConfig(strategy="zero", train_iters=3000, ...)

    # FSDP (experimental)
    engine_config = EngineConfig(strategy="fsdp", train_iters=3000, ...)

Launch
------

Once the ``strategy`` is configured in ``EngineConfig``, users can launch distributed
training as a normal PyTorch distributed training job via their native (elastic) launcher.
Depending on user's PyTorch version, some launchers may not be supported. Here, we provide
examples of using PyTorch's distributed launchers. More detailed instructions can be found
in PyTorch's official `Tutorial <https://pytorch.org/docs/stable/elastic/run.html>`_.

- PyTorch version :math:`\geq 1.10`

.. code::

    # simple
    torchrun train_script.py

    # detailed
    torchrun --rdzv_backend=c10d --rdzv_endpoint=MASTER_ADDR --nproc_per_node=NUM_GPUS --nnodes=NUM_NODES train_script.py


- PyTorch version :math:`< 1.10`

.. code:: 
    
    # simple
    python -m torch.distributed.launch --use_env train_script.py

    # detailed
    python -m torch.distributed.launch --nproc_per_node=NUM_GPUS --nnodes=NUM_NODES --node_rank=NODE --master_addr=MASTER_ADDR --master_port=MASTER_PORT train_script.py

Note that ``torch.distributed.launch`` requires users to include ``--local_rank`` in
``argparse`` as below:

.. code:: python

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    local_rank = args.local_rank



Behind the scene, Betty takes care of patching (1) module, (2) optimizer,
(3) data loader, (4) lr scheduler, (5) loss scaler (for mixed-precision training)
automatically for users.
