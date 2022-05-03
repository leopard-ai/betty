# Data Reweighting with Meta-Weight Net

We re-implemented the data reweighting algorithm from [Meta-Weight-Net: Learning an Explicit Mapping For Sample Weighting](https://arxiv.org/abs/1902.07379)


### Differences
While the original paper experiments with long-tailed/noisy CIFAR-10/100 image classification tasks, we only report the results for the long-tailed CIFAR-10 image classification task.
Interested users can easily run other tasks following below scripts.

Another difference from the original implementation to notice is the hypergradient calculation method. While the original implementation adopted the MAML-basd one-step gradient approach to calculate hypergradient, our implementation adopted the DARTS variant method with 5 gradient steps. Again, our framework supports a variety of hypergradient calculation methods with one line change in Config.

## Environment
Our code is developed/tested on:

- Python 3.8.10
- pytorch 1.10
- torchvision 1.11

## Scripts
ResNet32 on CIFAR-10 with the imbalance ratio of 50:
```
python main.py --imbalanced_factor 50
```
ResNet32 on CIFAR-10 with the 40% uniform noise:
```
python main.py --meta_lr 1e-3 --meta_weight_decay 1e-4 --corruption_type uniform --corruption_ratio 0.4
```

## Results
We present the long-tailed CIFAR-10 image classification in the below table.
|          | Imbalanced factor 200 | Imbalanced factor 100 | Imbalanced factor 50 | GPU Memory |
|----------|:---------------------:|:---------------------:|:--------------------:|:----------:|
| Original |         68.91%        |         75.21%        |        80.06%        |   2250MiB  |
| Ours     |       **75.56%**      |       **77.73%**      |      **80.26%**      | **2051MiB**|


## Acknowledgements
Our code is heavily built upon
- https://github.com/ShiYunyi/Meta-Weight-Net_Code-Optimization
- https://github.com/xjtushujun/meta-weight-net