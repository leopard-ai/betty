# NAS Augmented Image Captioning (IUC)

This the Betty re-implementation of
[Image Understanding by Captioning with Differentiable Architecture Search](https://dl.acm.org/doi/pdf/10.1145/3503161.3548150)
at ***ACM MM'22***.

## Abstract

In image captioning, models have encoder-decoder architectures,
where the encoders take the input images, produce embeddings,
and feed them into the decoders to generate textual descriptions.
Designing a proper image captioning encoder-decoder architecture
manually is a difficult challenge due to the complexity of recognizing
the critical objects of the input images and their relationships
to generate caption descriptions. To address this issue, we propose
a three-level optimization method that employs differentiable
architecture search strategies to seek the most suitable architecture
for image captioning automatically. Our optimization framework
involves three stages, which are performed end-to-end. In the first
stage, an image captioning model learns and updates the weights
of its encoder and decoder to create image captions. At the next
stage, the trained encoder-decoder generates a pseudo image
captioning dataset from unlabeled images, and the predictive mode
trains on the generated dataset to update its weights. Finally, the
trained model validates its performance on the validation set and
updates the encoder-decoder architecture by minimizing the validation
loss. Experiments and studies on the COCO image captions
datasets demonstrate that our method performs significantly better
than the baselines and can achieve state-of-the-art results in image
understanding tasks.

## Architecture Search

```python train_search_IUC.py```

## Results

|                        | BLEU-4 | CIDEr |
|------------------------|--------|-------|
| LSTM (baseline)        | 29.6   | 94.0  |
| AutoCaption (baseline) | 39.2   | 125.2 |
| IUC (official)         | 40.0   | 131.1 |
| IUC (Betty)            | 39.8   | 131.1 |
