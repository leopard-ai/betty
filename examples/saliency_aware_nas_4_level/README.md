# Saliency-Aware Neural Architecture Search (SANAS)

This the Betty re-implementation of
[Saliency-Aware Neural Architecture Search](https://openreview.net/pdf?id=Ho6oWAslz5L)
at ***NeurIPS'2022***.

## Abstract

Existing NAS methods ignore the fact that different input data elements (e.g., image pixels)
have different importance (or saliency) in determining the prediction outcome.
They treat all data elements as being equally important and therefore lead to suboptimal
performance. To address this problem, we propose an end-to-end framework
which dynamically detects saliency of input data, reweights data using saliency
maps, and searches architectures on saliency-reweighted data. Our framework is
based on four-level optimization, which performs four learning stages in a unified
way. At the first stage, a model is trained with its architecture tentatively fixed. At
the second stage, saliency maps are generated using the trained model. At the third
stage, the model is retrained on saliency-reweighted data. At the fourth stage, the
model is evaluated on a validation set and the architecture is updated by minimizing
the validation loss.

## Training

You can change the model search using ```--darts_type```:

```bash
python train_search_sanas.py
```

## Evaluation

```bash
python train.py --auxiliary --cutout
```

## Results

| CIFAR-100        | Test Accuracy |
|------------------|---------------|
| DARTS (baseline) | 79.42         |
| SANAS (official) | 83.20         |
| SANAS (Betty)    | 82.81         |
