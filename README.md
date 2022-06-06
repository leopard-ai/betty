# Betty

`betty` is a PyTorch-based automatic differentiation library for *multielvel optimization*,
a generalization of meta-learning or bilevel optimization.

It can be used as a unified programming interface for a wide range of multilevel optimization
applications, including meta-learning, hyperparameter optimization, neural architecture search,
reinforcement learning, etc.

## Why Betty?
An implementation of gradient-based multilevel optimization is notoriously complicated.
It requires (1) approximating gradient for upper-level problems using iterative/implicit
differentiation, and (2) writing nested for loops to handle the hierarchical dependency between
multiple problems. Thus, implementing multilevel optimization applications oftentimes requires
expertise in both programming and mathematics, raising a major barrier to research in this field.

`betty` abstracts away the above implementation difficulties by re-interpreting multilevel
optimization from the dataflow graph perspective. More specifically, the dataflow graph
interpretation enables
- **automatic differentiation** that hides low-level implementation details of gradient
calculation/approximation in multilevel optimization behind the API as autodiff libraries for neural
networks do (e.g. PyTorch, Tensorflow).
- **easy-to-use programming interface** that allows users to focus on implementing each level
optimization problem independently without having to worry about hierarchical interactions between
them. Particularly, our interface for implementing each level problem is very similar with
PyTorch's interface for implementing `torch.nn.Module`.


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


#### Engine


## Features
#### Gradient Approximation Methods
- Implicit Differentiation
  - Finite Difference ([DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055))
  - Neumann Series ([Optimizing Millions of Hyperparameters by Implicit Differentiation](http://proceedings.mlr.press/v108/lorraine20a/lorraine20a.pdf))
  - Conjugate Gradient (ref: [Meta-Learning with Implicit Gradients](https://proceedings.neurips.cc/paper/2019/file/072b030ba126b2f4b2374f342be9ed44-Paper.pdf))
- Iterative Differentiation
  - Reverse-mode Automatic Differentiation ([Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400))


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