<h3 align="center">
  Betty
</h3>
<p align="center">
  An automatic differentiation library for multilevel optimization<br>
  <a href="https://www.google.com/">Tutorial</a> |
  <a href="https://www.google.com/">Docs</a> |
  <a href="https://www.google.com/">Examples</a> |
  <a href="https://www.google.com/">CASL Project</a>
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
### Problem
#### Basics
Each level problem can be defined with 7 components: (1) module, (2) optimizer, (3) data loader,
(4) loss function, (5) problem configuration, (6) name, and (7) other optional components (e.g.
learning rate scheduler). (4) loss function can be defined with the `training_step` method, while
all other components can be provided through the class constructor. For example, image
classification problem can be defined as follows:
```python
from betty.problems import ImplicitProblem
from betty.configs import Config

# set up module, optimizer, data loader (i.e. (1)-(3))
cls_module, cls_optimizer, cls_data_loader = setup_classification()

class Classifier(ImplicitProblem):
    # set up loss function
    def training_step(self, batch):
        inputs, labels = batch
        outputs = self.module(inputs)
        loss = F.cross_entropy(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels.long()).float().mean().item() * 100

        # Returned dict will be automatically logged with the logging tool (e.g. TensorBoard)
        return {'loss': loss, 'acc': acc}

# set up problem configuration
cls_config = Config(type='darts', step=1, log_step=10, fp16=False, retain_graph=True)

# Classifier problem class instantiation
cls_prob = Classifier(name='classifier',
                      module=cls_module,
                      optimizer=cls_optimizer,
                      train_data_loader=cls_data_loader,
                      config=cls_config)
```

#### Interaction with other problems
In MLO, each problem often needs to access modules from other problems to define its loss function.
This can be achieved by using the `name` attributed

```python
class HPO(ImplicitProblem):
    def training_step(self, batch):
        # set up hyperparameter optimization loss
        ...

# HPO problem class instantiation
hpo_prob = HPO(name='hpo', module=...)

class Classifier(ImplicitProblem):
    def training_step(self, batch):
        inputs, labels = batch
        outputs = self.module(inputs)
        loss = F.cross_entropy(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels.long()).float().mean().item() * 100
        
        # accessing weight decay hyperparameter from another problem HPO can be achieved by its
        # name 'hpo'
        weight_decay = self.hpo()
        reg_loss = weight_decay * sum([p.norm().pow(2) for p in self.module.parameters()])
        
        return {'loss': loss + reg_loss, 'acc': acc}

cls_prob = Classifier(name='classifier', module=...)
```
#### Engine
```python
from betty import Engine
```

## Features
### Gradient Approximation Methods
- Implicit Differentiation
  - Finite Difference ([DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055))
  - Neumann Series ([Optimizing Millions of Hyperparameters by Implicit Differentiation](http://proceedings.mlr.press/v108/lorraine20a/lorraine20a.pdf))
  - Conjugate Gradient ([Meta-Learning with Implicit Gradients](https://proceedings.neurips.cc/paper/2019/file/072b030ba126b2f4b2374f342be9ed44-Paper.pdf))
- Iterative Differentiation
  - Reverse-mode Automatic Differentiation ([Model-Agnostic Meta-Learning (MAML)](https://arxiv.org/abs/1703.03400))


### Training
- Gradient accumulation
- FP16 training (unstable)
- Distributed data-parallel training (TODO)

### Logging
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
