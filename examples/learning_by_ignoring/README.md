# Learning by Ignoring
---
## Introduction


## Environment
Our code is developed/tested on:

- Python 3.8.10
- pytorch 1.10
- torchvision 1.11

## Scripts
Baseline:
```
python main.py --gpu=0 --source_domain=Cl --target_domain=Ar --lam=7e-3 --baseline
```
Learning by Ignoring:
```
python main.py --gpu=0 --source_domain=Cl --target_domain=Ar --lam=7e-3
```

## Results
We present the long-tailed CIFAR-10 image classification in the below table.

|            | Cl-Ar  | Ar-Pr  | Pr-Rw  | Rw-Cl  |
|------------|--------|--------|--------|--------|
| Baseline   | 65.43% | 88.06% | 76.98% | 69.49% |
| LBI (Ours) | **66.46%** | **88.29%** | **77.09%** | **70.86%** |