Distributed Training
====================

Despite the popularity of meta-learning, research in meta-learning has largely been limited to
small-scale setups due to the memory and computation bottlenecks. In detail, meta-learning
oftentimes requires second-order gradient information (i.e., Hessian) and/or multiple
forward/backward computations.

Distributed training emerges as a natural solution to mitigate above issues. To keep up with the
recent trend of large foundation models, Betty supports distributed training for meta-learning in
a seamless way. More specifically, users can enable distributed training by:


#. Step 1. Modifying `EngineConfig`.

Users can simply set `distributed=True` to enable distributed training with Betty.

.. code:: python

    engine_config = EngineConfig(distributed=True, train_iters=3000, ...)

#. Step 2. Launching with `torch.distributed.launch`

Users can follow the standard PyTorch distributed training launch script to execute distributed
training. Note that PyTorch distributed training requires users to add `local_rank` in Argument.

.. code:: 

    python -m torch.distributed.launch --nproc_per_node=NUM_GPUS --nnodes=NUM_NODES --node_rank=NODE --master_addr=MASTER_ADDR --master_port=MASTER_PORT your_training_scripts


Behind the scene, Betty takes care of patching data loader to support distributed sampler,
gradient synchronization, parameter synchronization, etc.
