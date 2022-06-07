<h3 align="center">
  Betty
</h3>
<p align="center">
  An automatic differentiation library for multielvel optimization or generalized meta-learning. 
</p>

### Installation
**pip** (general use)

```bash
pip install betty
```

**From source** (developers)

```bash
pip install -e .
```

## What is Betty?
It can be used as a unified programming interface for a wide range of multilevel optimization
applications, including meta-learning, hyperparameter optimization, neural architecture search,
reinforcement learning, etc.

## Why Betty?
An implementation of gradient-based multilevel optimization or meta-learning is notoriously
complicated. It requires (1) approximating gradient for upper-level problems using
iterative/implicit differentiation, and (2) writing nested for-loops to handle the hierarchical
dependency between multiple levels.

With `betty`, users do not need to care about any of the above issues, as the API will handle all
the implementation intricacies internally. Instead, users only need to do two things to implement
multilevel optimization programs:
1. Define each optimization problem with the `Problem` class. The programming interface of the
`Problem` class is very similar with that of PyTorch Lightning's `LightningModel`.
1. Define the hierarchical problem dependency with the `Engine` class.

Users interested in detailed internal mechanisms and software architectures are encouraged to
read our [paper](.) and [documentation](.).


## Installation
The stable version of `betty` can be easily installed with [PyPI](https://pypi.org/) (or pip).
```bash
pip install betty
```

#### From Source
For users interested in the latest (but unstable) version, or contributing to the development of
 `betty`, we provide an option to install the library from source.
```bash
pip install -r requirements.txt
pip install -e .
```

#### Requirments
The dependency of `betty` includes:
- Python >= 3.6
- PyTorch >= 1.10
- tensorboard
- functorch

## Examples
#### Problem
```python
import betty
```

#### Engine
```python
import betty
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
If you use `betty` in your research, please cite our arxiv paper with the following Bibtex entry.
```
@article{choe2022betty,
  title="Betty: An Automatic Differentiation Library for Multilevel Optimization",
  author="Choe, Sang Keun and Neiswanger, Willie and Xie, Pengtao and Xing, Eric",
  year=2022,
  url="http://arxiv.org/abs/2008.12284"
}
```

## License
`betty` is licensed under the [MIT License](LICENSE).