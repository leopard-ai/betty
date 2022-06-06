# Betty

`betty` is a PyTorch-based automatic differentiation library for *multielvel optimization*, a generealization of meta-learning or bilevel optimization.

The goal of `betty` is to provide a unified programming interface for a wide range of multilevel optimization applications, including meta-learning, hyperparameter optimization, neural architecture search, reinforcement learning, etc.

## Introduction


## Installation
The stable version of `betty` can be easily installed with [PyPI](https://pypi.org/) (or pip).
```
pip install betty
```

#### From Source
For users interested in the latest (but unstable) version, or contributing to the development of `betty`, we provide an option to install the library from source.
```
pip install -r requirements.txt
pip install -e .
```

#### Requirments
    - Python >= 3.6
    - PyTorch >= 1.10
    - Tensorboard
    - Functorch
