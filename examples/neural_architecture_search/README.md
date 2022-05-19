# Differentiable Neural Architecture Search (DARTS)

## Introduction
We re-implement [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) where
they developed the continuous relaxation of the discrete architecture representation so as to allow 
efficient search of the architecture using gradient descent.

### Differences
While the original paper performs a finite difference method on the initial network weight, 
we perform it on the unrolled network weight.
This is because we view DARTS' hypergradient calculatation method from the implicit differentitation
perspective, where the second-order derivative is calculated based on the unrolled weight.
Interested users can refer to 
[Optimizing Millions of Hyperparameters by Implicit Differentiation](https://arxiv.org/pdf/1911.02590)
.

This also allows us to unroll the inner loop more than one iteration as opposed to one-step unrolled
learning used in the original paper. We present the result for different unrolling steps in the 
below Results Section.


## Environment
Our code is developed/tested on:

- Python 3.8.10
- pytorch 1.10
- torchvision 1.11

## Scripts
Learning the architecture along with the network parameters:
```
python train_search.py --unroll_steps=1
```
Training the network with the *learned* architecture:
```
python train.py --cutout --auxiliary
```

## Results
We present the CIFAR-10 image classification results in the below table.
|                      | Test Acc. | Params (M) |
|----------------------|-----------|------------|
| DARTS (original)     | 97.24%    | 3.3        |
| DARTS (ours, step=1) | 97.35%    | 3.7        |
| DARTS (ours, step=3) | 97.22%    | 3.2        |

## Acknowledgements
Our code is heavily built upon:
- https://github.com/quark0/darts
- https://github.com/dragen1860/DARTS-PyTorch
