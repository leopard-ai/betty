<h3 align="center">
  Betty
</h3>

```bash
pip install betty
```

## What is Betty?
Betty is an automatic differentiation library for *multilevel optimization (MLO)* and/or
*generalized meta-learning*.

## Why Betty?
An implementation of gradient-based multilevel optimization or meta-learning is notoriously
complicated. For example, it requires approximating gradient for upper-level problems using
iterative/implicit differentiation, and writing nested for-loops to handle the hierarchical
dependency between multiple levels.

Good news is that Betty hides most of such implementation intricacies behind the API while allowing
users to write only high-level code. Now, users simply need to do two things to implement
multilevel optimization programs:
1. Define each optimization problem with the `Problem` class. The programming interface of the
`Problem` class is very similar with that of PyTorch Lightning's `LightningModel`.
2. Define the hierarchical problem dependency with the `Engine` class.

As a result, Betty allows an easy-to-use, modular and unified programming interface for various
MLO applications including meta-learning, hyperparameter optimization, neural architecture search,
reinforcement learning, etc.

## Examples
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
- tensorboard
- wandb

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

## License
Betty is licensed under the [MIT License](LICENSE).