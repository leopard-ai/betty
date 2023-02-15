# Meta-Learning with Implicit Gradients

In this example, we re-implement
[Meta-Learning with Implicit Gradients](https://arxiv.org/abs/1909.04630)

## Setup

We rely on [learn2learn](https://learn2learn.net/) for the data loading part. Please install
`learn2learn` before running the main training script.

```bash
pip install learn2learn
```

## Script

### Omniglot
- Shots: [1, 5]
- Ways: [5, 20]

```bash
bash omniglot.sh
```

### Mini-ImageNet
- Shots: [1, 5]
- Ways: 5

```bash
bash mini_imagenet.sh
```

## Results

As our goal is to demonstrate how to implement (i)MAML with Betty, we didn't do
extensive hyperparameter tuning in our experiments. Also, we used DARTS algorithm
for hypergradient calculation, which achieves better computation/memory efficiency
than conjugate gradient (CG) algorithm in the original paper. Further performance
improvement can be achieved by trying out different hypergradient algorithms and
hyperparameters.

### Omniglot

|               | 5-way 1-shot | 5-way 5-shot | 20-way 1-shot | 20-way 5-shot |
|---------------|--------------|--------------|---------------|---------------|
| MAML          | 98.7%        | 99.9%        | 95.8%         | 98.9%         |
| FOMAML        | 98.3%        | 99.2%        | 89.4%         | 97.9%         |
| Reptile       | 97.68%       | 99.48%       | 89.43%        | 97.12%        |
| iMAML (orig)  | 99.16%       | 99.67%       | 94.46%        | 98.69%        |
| iMAML (Betty) | 98.68%       | 99.58%       | 92.90%        | 98.04%        |

### Mini-ImageNet
To be added.
