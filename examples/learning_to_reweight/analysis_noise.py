import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import *
from data import *
from utils import *

from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig


parser = argparse.ArgumentParser(description="Meta_Weight_Net")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--precision", type=str, default="fp32")
parser.add_argument("--strategy", type=str, default="default")
parser.add_argument("--rollback", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--meta_net_hidden_size", type=int, default=100)
parser.add_argument("--meta_net_num_layers", type=int, default=1)

parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--dampening", type=float, default=0.0)
parser.add_argument("--nesterov", type=bool, default=False)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--meta_lr", type=float, default=1e-5)
parser.add_argument("--meta_weight_decay", type=float, default=0.0)

parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--num_meta", type=int, default=1000)
parser.add_argument("--imbalanced_factor", type=int, default=None)
parser.add_argument("--corruption_type", type=str, default=None)
parser.add_argument("--corruption_ratio", type=float, default=0.0)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--max_epoch", type=int, default=120)

parser.add_argument("--meta_interval", type=int, default=1)
parser.add_argument("--paint_interval", type=int, default=20)

args = parser.parse_args()
print(args)
set_seed(args.seed)

resume_indexes = torch.load("train_index.pt")
resume_labels = torch.load("train_label.pt")
orig_labels = torch.load("orig_label.pt")

(
    train_dataloader,
    meta_dataloader,
    test_dataloader,
    imbalanced_num_list,
) = build_dataloader(
    seed=args.seed,
    dataset=args.dataset,
    num_meta_total=args.num_meta,
    imbalanced_factor=args.imbalanced_factor,
    corruption_type=args.corruption_type,
    corruption_ratio=args.corruption_ratio,
    batch_size=args.batch_size,
    resume_idxes=resume_indexes,
    resume_labels=resume_labels,
    analysis=True,
)

net = ResNet32(args.dataset == "cifar10" and 10 or 100)
meta_net = MLP(
    hidden_size=args.meta_net_hidden_size, num_layers=args.meta_net_num_layers
)
net.cuda()
meta_net.cuda()

correct_idx = []
noise_idx = []
for idx, (t, f) in enumerate(zip(orig_labels, resume_labels)):
    if t == f:
        correct_idx.append(idx)
    else:
        noise_idx.append(idx)

print("noise ratio:", 1 - len(correct_idx) / len(orig_labels))

checkpoints = [
    500,
    1000,
    1500,
    2000,
    2500,
    3000,
    3500,
    4000,
    4500,
    5000,
    5500,
    6000,
    6500,
    7000,
    7500,
    8000,
    8500,
    9000,
]
# checkpoints = [5000,6000,6500,7000,7500]
sample_weights = 0
with torch.no_grad():
    for ckpt in checkpoints:
        net.eval()
        meta_net.eval()

        net.load_state_dict(torch.load(f"{args.dataset}/net_{ckpt}.pt")["module"])
        meta_net.load_state_dict(
            torch.load(f"{args.dataset}/meta_net_{ckpt}.pt")["module"]
        )

        importance_weight = np.zeros((10))
        frequency = np.zeros((10))
        sample_weight = []
        for data, label in train_dataloader:
            data, label = data.cuda(), label.cuda()

            out = net(data)
            loss = F.cross_entropy(out, label.long(), reduction="none")
            loss_vector = torch.reshape(loss, (-1, 1))
            weight = meta_net(loss_vector).squeeze()

            sample_weight.append(weight.cpu().numpy())

        print("\n==================================================")
        print("checkpoint:", ckpt)

        sample_weight = np.concatenate(sample_weight)
        sample_weights += sample_weight / len(checkpoints)

        noise_weight = sample_weight[noise_idx]
        correct_weight = sample_weight[correct_idx]

        print("correct weight:", np.mean(correct_weight), len(correct_weight))
        print("noise weight:", np.mean(noise_weight), len(noise_weight))

print("\n==================================================")
final_noise_weight = sample_weights[noise_idx]
final_correct_weight = sample_weights[correct_idx]
print("final correct weight:", np.mean(final_correct_weight))
print("final noise weight:", np.mean(final_noise_weight))
n, bins, patches = plt.hist(
    x=final_correct_weight, bins="auto", color="r", alpha=0.7, rwidth=0.85
)
n, bins, patches = plt.hist(
    x=final_noise_weight, bins=bins, color="b", alpha=0.7, rwidth=0.85
)
plt.xlabel("weight")
plt.ylabel("frequency")
plt.show()
