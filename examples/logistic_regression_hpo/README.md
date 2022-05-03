# Hyperparameter Optimization for Logistic Regression

As a toy example, we implement hyperparameter optimization for logistic regression where we optimize
weight decay values for *all* regression parameters.
We verified that a variety of hypergradient calculation methods
(*e.g.,* MAML, DARTS, Neumann, Conjugate gradient) all lead to the same performance.

## Acknowledgements
Our code is built upon the logistic regression example in
[hypertorch](https://github.com/prolearner/hypertorch).