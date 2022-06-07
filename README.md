<h3 align="center">
  Betty
</h3>
<p align="center">
  An automatic differentiation library for multilevel optimization
</p>

```bash
pip install betty
```

## What is Betty?
Betty is a [PyTorch](https://pytorch.org) library for multilevel optimization (MLO) that provides
a unified programming interface for various applications including meta-learning, hyperparameter
optimization, neural architecture search, data reweighting, reinforcement learning, etc.

## Why Betty?
An implementation of multilevel optimization is notoriously complicated. For example, it
requires approximating gradient using iterative/implicit differentiation, and writing nested
for-loops to handle the hierarchical dependency between multiple levels.

Good news is that Betty abstracts away such low-level implementation details behind the API, while
allowing users to write only high-level code. Now, users simply need to do two things to implement
any MLO programs:
1. Define each level optimization problem with the [Problem](#problem) class. 
2. Define the hierarchical problem dependency with the [Engine](#engine) class.

## How to use Betty?
#### Problem
```python
from betty.problems import ImplicitProblem
```

#### Engine
```python
from betty import Engine
```

## Features
#### Gradient Approximation Methods
- Implicit Differentiation
  - Finite Difference ([DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055))
  - Neumann Series ([Optimizing Millions of Hyperparameters by Implicit Differentiation](http://proceedings.mlr.press/v108/lorraine20a/lorraine20a.pdf))
  - Conjugate Gradient ([Meta-Learning with Implicit Gradients](https://proceedings.neurips.cc/paper/2019/file/072b030ba126b2f4b2374f342be9ed44-Paper.pdf))
- Iterative Differentiation
  - Reverse-mode Automatic Differentiation ([Model-Agnostic Meta-Learning (MAML)](https://arxiv.org/abs/1703.03400))


#### Training
- Gradient accumulation
- FP16 training (unstable)
- Distributed data-parallel training (TODO)

#### Logging
- [(PyTorch) TensorBoard](https://pytorch.org/docs/stable/tensorboard.html)
- [wandb](https://github.com/wandb/client)

## Citation
If you use Betty in your research, please cite our arxiv paper with the following Bibtex entry.
```
@article{choe2022betty,
  title="Betty: An Automatic Differentiation Library for Multilevel Optimization",
  author="Choe, Sang Keun and Neiswanger, Willie and Xie, Pengtao and Xing, Eric",
  year=2022,
  url="http://arxiv.org/abs/2008.12284"
}
```

## Contributing
To be updated.

## License
Betty is licensed under the [MIT License](LICENSE).
