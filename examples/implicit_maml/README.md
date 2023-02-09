# Meta-Learning with Implicit Gradients

In this example, we re-implemented
[Meta-Learning with Implicit Gradients](https://arxiv.org/abs/1909.04630)

## Setup

We rely on [TorchMeta](https://github.com/tristandeleu/pytorch-meta) for the data loading.
However, (pip) installing TorchMeta may encounter a PyTorch version incompatibility issue.
We avoided this issue by:

1. Download the source code from their [repo](https://github.com/tristandeleu/pytorch-meta)
2. Remove the torch and torchvision dependency in `setup.py`
3. Replace functions imported in this
[line](https://github.com/tristandeleu/pytorch-meta/blob/master/torchmeta/datasets/utils.py#L39)
by following this
[StackOverflow](https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039).

## Script

You can run the iMAML experiment with the following script:

```bash
python main.py --inner_steps 5
```

## Warning

Our implementation currently doesn't match the accuracy of the original paper.
We are actively investigating the issue, and update the repo as soon as possible.

## Acknowledgement

We followed the CNN architecture from [learn2learn](https://learn2learn.net).