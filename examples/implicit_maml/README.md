# Meta-Learning with Implicit Gradients

In this example, we re-implement
[Meta-Learning with Implicit Gradients](https://arxiv.org/abs/1909.04630)

## Setup

We rely on [learn2learn](https://learn2learn.net/) for the data loading part. Please install
`learn2learn` before running the main training script.

```bash
pip install learn2learn
```

## Script

You can run the iMAML experiment with the following script:

```bash
python main.py --task omniglot --shots 1 --ways 5 --inner_steps 5
```

## Results

To be added.
