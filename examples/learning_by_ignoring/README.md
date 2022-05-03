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
Run all experiments:
```
bash run.sh
```

## Results
We present the long-tailed CIFAR-10 image classification in the below table.

|            | Cl-Ar  | Ar-Pr  | Pr-Rw  | Rw-Cl  |
|------------|--------|--------|--------|--------|
| Baseline   | 65.63% | 87.35% | 77.88% | 68.00% |
| LBI (Ours) | **66.87%** | **88.88%** | **78.77%** | **70.17%** |