import argparse
from pathlib import Path

from tqdm import tqdm
import os
import json
import numpy
import torch
import random
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from model import resnet18, resnet50, MLP
from dataset import ImageNet


parser = argparse.ArgumentParser(description="Filter stage")
parser.add_argument(
    "--seed", type=int, default=42, metavar="S", help="random seed (default: 1)"
)
parser.add_argument("--layers", type=int, default=50)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--data-dir", default=Path("./data"), type=Path)
parser.add_argument("--filter_ratio", type=float, default=0.5)
parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    help="input batch size for inference (default: 256)",
)

parser.add_argument("--interpolation", default="bilinear", type=str)
parser.add_argument("--val-resize-size", default=256, type=int)
parser.add_argument("--val-crop-size", default=224, type=int)
parser.add_argument("--train-crop-size", default=224, type=int)
parser.add_argument("--imagenet-classes", default="metadata/imagenet_classes.json")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--checkpoint_directory", type=str, default=".")
parser.add_argument("--desc", type=str, default="130_150k")

parser.add_argument("--random", action="store_true")

args = parser.parse_args()
random.seed(args.seed)
numpy.random.seed(args.seed)
torch.manual_seed(args.seed)

# Data loading
with open(args.imagenet_classes) as f:
    classes = list(json.load(f)["classes"])

train_dataset = ImageNet(
    dataset_file=args.data_dir,
    sample_set="train",
    classes=classes,
    transform_type="val",
    args=args,
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)

average_list = [130104, 135108, 140112, 145116, 150120]

desc = args.desc
os.makedirs(os.path.join(args.checkpoint_directory, desc), exist_ok=True)

sorted_idx = numpy.random.permutation(len(train_dataset))
sorted_weight = []
weights_total = []

if not args.random:
    print("MetaWeight")

    for idx in average_list:
        if args.layers == 18:
            classifier = resnet18()
        else:
            classifier = resnet50()

        classifier.load_state_dict(
            torch.load("{}/cls_{}.pt".format(args.checkpoint_directory, idx))
        )
        classifier.to(args.device)
        classifier.eval()

        mwn = MLP(2, 100, 1)
        mwn.load_state_dict(
            torch.load("{}/mwn_{}.pt".format(args.checkpoint_directory, idx))
        )
        mwn.to(args.device)
        cur_iter = idx
        mwn.eval()

        weights = []
        with torch.no_grad():
            for batch in tqdm(train_dataloader):
                inputs, labels = batch
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                outputs, ema_outputs = classifier(inputs)
                loss = torch.nn.functional.cross_entropy(
                    outputs, labels.long(), reduction="none"
                )
                loss = torch.reshape(loss, (-1, 1))
                ema_prob = torch.nn.functional.softmax(ema_outputs, dim=-1)
                ema_loss = torch.sum(
                    -torch.nn.functional.log_softmax(outputs, dim=-1) * ema_prob, dim=-1
                )
                ema_loss = torch.reshape(ema_loss, (-1, 1))
                weight = mwn(
                    torch.cat([loss.detach(), ema_loss.detach()], dim=1), test=True
                )
                weights.extend(weight.squeeze().cpu().numpy())
        weights_total.append(weights)

    avg_weights = numpy.array(weights_total).mean(axis=0)
    sorted_idx = numpy.argsort(avg_weights)
    sorted_idx = sorted_idx[::-1]
    sorted_weight = [avg_weights[i] for i in sorted_idx]
    print("Top 100 scores:", sorted_weight[:100])
    print("Bottom 100 scores:", sorted_weight[-100:])

torch.save(
    sorted_idx, "{}/sorted_idx.pt".format(os.path.join(args.checkpoint_directory, desc))
)
torch.save(
    sorted_weight,
    "{}/sorted_weight.pt".format(os.path.join(args.checkpoint_directory, desc)),
)
torch.save(
    weights_total,
    "{}/weights_total.pt".format(os.path.join(args.checkpoint_directory, desc)),
)
