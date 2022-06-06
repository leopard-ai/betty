# Data Reweighting for Pre-training

## Introduction
Pretraining/finetuning frameworks are getting popularized with the recent advancement in
self-supervised learning.
However, pretraining data are oftentimes from a different distribution than finetuning data,
which could potentially cause negative transfer.
To overcome this issue,
[Learning by Ignoring](https://arxiv.org/pdf/2012.14288.pdf) combines a data reweighting shceme with
pretraining/finetuning frameworks to automatically suppress the weight of pretraining data samples
that cause negative transfer while increase the weight of others.
The similar idea was also proposed in
[Meta-Learning to Improve Pre-TrainingMeta-learning](https://arxiv.org/abs/2111.01754)
(NeurIPS 2021).

## Dataset
OfficeHome dataset can be downloaded from [here](http://155.33.198.138/lbi_data.zip)
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
We present the result of Learning by Ignoring on the OfficeHome datset.

|            | Cl-Ar  | Ar-Pr  | Pr-Rw  | Rw-Cl  |
|------------|--------|--------|--------|--------|
| Baseline   | 65.63% | 87.35% | 77.88% | 68.00% |
| LBI (Ours) | **66.87%** | **88.88%** | **78.77%** | **70.17%** |