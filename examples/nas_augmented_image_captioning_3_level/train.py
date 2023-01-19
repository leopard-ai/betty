import os
import sys
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from model import NetworkCIFAR as Network
import genotypes
import utils

parser = argparse.ArgumentParser("cifar10")
parser.add_argument(
    "--data", type=str, default="../data", help="location of the data corpus"
)
parser.add_argument("--batchsz", type=int, default=96, help="batch size")
parser.add_argument("--lr", type=float, default=0.025, help="init learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--wd", type=float, default=3e-4, help="weight decay")
parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument("--epochs", type=int, default=600, help="num of training epochs")
parser.add_argument("--init_ch", type=int, default=36, help="num of init channels")
parser.add_argument("--layers", type=int, default=20, help="total number of layers")
parser.add_argument(
    "--model_path", type=str, default="saved_models", help="path to save the model"
)
parser.add_argument(
    "--auxiliary", action="store_true", default=False, help="use auxiliary tower"
)
parser.add_argument(
    "--auxiliary_weight", type=float, default=0.4, help="weight for auxiliary loss"
)
parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
parser.add_argument(
    "--drop_path_prob", type=float, default=0.2, help="drop path probability"
)
parser.add_argument(
    "--exp_path", type=str, default="exp/cifar10", help="experiment name"
)
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument(
    "--arch", type=str, default="DARTS", help="which architecture to use"
)
parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
args = parser.parse_args()


genotype = torch.load("genotype.t7")["genotype"]
model = Network(args.init_ch, 10, args.layers, args.auxiliary, genotype).cuda()

print(f"[*] Number of parameters: {utils.count_parameters_in_MB(model)}")

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(
    model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd
)

train_transform, valid_transform = utils.data_transforms_cifar10(args)
train_data = dset.CIFAR10(
    root=args.data, train=True, download=True, transform=train_transform
)
valid_data = dset.CIFAR10(
    root=args.data, train=False, download=True, transform=valid_transform
)

train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batchsz, shuffle=True, pin_memory=True, num_workers=2
)
valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batchsz, shuffle=False, pin_memory=True, num_workers=2
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))


def train(train_loader, model, criterion, optimizer):
    model.train()

    for x, target in train_loader:
        x = x.cuda()
        target = target.cuda(non_blocking=True)

        optimizer.zero_grad()
        logits, logits_aux = model(x)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()


def infer(valid_loader, model):
    model.eval()

    correct = 0
    total = 0
    for x, target in valid_loader:
        x = x.cuda()
        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            logits, _ = model(x)
            correct += (logits.argmax(dim=1) == target).float().sum().item()
            total += x.size(0)

    acc = correct / total * 100
    print(f"[*] Validation Acc.: {acc}")


for epoch in range(args.epochs):
    scheduler.step()
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train(train_queue, model, criterion, optimizer)
    infer(valid_queue, model)
