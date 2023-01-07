# Data Reweighting (Meta-Weight Net) with BERT

The original [Meta-Weight-Net](https://arxiv.org/abs/1902.07379) is only
tested with relatively small models like ResNet. In this example, we try
to scale up the model from ResNet to BERT with Betty's various systems
support.


## Setup
- Model: (pre-trained) BERT-base from Hugging Face
- Dataset: SST-2 benchmark. We artificially injected class imbalance 
via `args.imbalance_factor`

## Scripts
- No meta-learning (baseline)

```
python main.py --baseline
```

- Meta-learning (Single GPU)

```
python main.py --baseline
```

- Meta-learning (Single GPU + mixed-precision)

```
python main.py --baseline --fp16
```

- Meta-learning (Multi GPU)

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --fp16 --strategy distributed
```

- Meta-learning (Multi GPU + ZeRO optimizer)

```
torchrun --standalone --nnodes=1 --nproc_per_node=2 main.py --fp16 --strategy zero
```

## Results
We present the long-tailed CIFAR-10 image classification in the below table.
|                       | Imbalanced factor 20 | GPU Memory |
|-----------------------|:--------------------:|:----------:|
| Baseline              |         68.91%       |   2250MiB  |
| Single GPU            |       **75.56%**     | **2051MiB**|
| + mixed-precision     |       **75.56%**     | **2051MiB**|
| + distributed         |       **75.56%**     | **2051MiB**|
| + zero                |       **75.56%**     | **2051MiB**|


## Acknowledgements
We modified the data loading code from
https://github.com/YJiangcm/SST-2-sentiment-analysis.