import os, time, glob
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem

from model_search import Network, Architecture
import utils


parser = argparse.ArgumentParser("cifar")
parser.add_argument(
    "--data", type=str, default="../data", help="location of the data corpus"
)
parser.add_argument("--batchsz", type=int, default=64, help="batch size")
parser.add_argument("--lr", type=float, default=0.025, help="init learning rate")
parser.add_argument("--lr_min", type=float, default=0.001, help="min learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--wd", type=float, default=3e-4, help="weight decay")
parser.add_argument("--report_freq", type=int, default=100, help="report frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument("--epochs", type=int, default=50, help="num of training epochs")
parser.add_argument("--init_ch", type=int, default=16, help="num of init channels")
parser.add_argument("--layers", type=int, default=8, help="total number of layers")
parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
parser.add_argument("--cutout_len", type=int, default=16, help="cutout length")
parser.add_argument(
    "--drop_path_prob", type=float, default=0.3, help="drop path probability"
)
parser.add_argument(
    "--train_portion", type=float, default=0.5, help="portion of training/val splitting"
)
parser.add_argument(
    "--arch_lr", type=float, default=3e-4, help="learning rate for arch encoding"
)
parser.add_argument(
    "--arch_wd", type=float, default=1e-3, help="weight decay for arch encoding"
)
parser.add_argument("--arch_steps", type=int, default=4, help="architecture steps")
parser.add_argument("--unroll_steps", type=int, default=1, help="unrolling steps")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda")

train_transform, valid_transform = utils.data_transforms_cifar10(args)
train_data = dset.CIFAR10(
    root=args.data, train=True, download=True, transform=train_transform
)
valid_data = dset.CIFAR10(
    root=args.data, train=False, download=True, transform=valid_transform
)

test_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batchsz, shuffle=False, pin_memory=True, num_workers=2
)

num_train = len(train_data)  # 50000
indices = list(range(num_train))
split = int(np.floor(args.train_portion * num_train))

train_iters = int(
    args.epochs
    * (num_train * args.train_portion // args.batchsz + 1)
    * args.unroll_steps
)

arch_net = Architecture(steps=args.arch_steps)
arch_optimizer = optim.Adam(
    arch_net.parameters(),
    lr=args.arch_lr,
    betas=(0.5, 0.999),
    weight_decay=args.arch_wd,
)
arch_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batchsz,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
    pin_memory=True,
    num_workers=2,
)

criterion = nn.CrossEntropyLoss().to(device)
classifier_net = Network(
    args.init_ch, 10, args.layers, criterion, steps=args.arch_steps
)
classifier_optimizer = optim.SGD(
    classifier_net.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.wd,
)
classifier_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    classifier_optimizer, float(train_iters // args.unroll_steps), eta_min=args.lr_min
)
classifier_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batchsz,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    pin_memory=True,
    num_workers=2,
)


class Arch(ImplicitProblem):
    def training_step(self, batch):
        x, target = batch
        alphas = self.forward()
        loss = self.classifier.module.loss(x, alphas, target)

        return loss


class Classifier(ImplicitProblem):
    def training_step(self, batch):
        x, target = batch
        alphas = self.arch()
        loss = self.module.loss(x, alphas, target)

        return loss


class NASEngine(Engine):
    @torch.no_grad()
    def validation(self):
        corrects = 0
        total = 0
        for x, target in test_queue:
            x, target = x.to(device), target.to(device, non_blocking=True)
            alphas = self.arch()
            _, correct = self.classifier.module.loss(x, alphas, target, acc=True)
            corrects += correct
            total += x.size(0)
        acc = corrects / total

        alphas = self.arch()
        torch.save({"genotype": self.classifier.module.genotype(alphas)}, "genotype.t7")
        return {"acc": acc}


outer_config = Config(retain_graph=True)
inner_config = Config(type="darts", unroll_steps=args.unroll_steps)
engine_config = EngineConfig(
    valid_step=args.report_freq * args.unroll_steps,
    train_iters=train_iters,
    roll_back=True,
)
outer = Arch(
    name="arch",
    module=arch_net,
    optimizer=arch_optimizer,
    train_data_loader=arch_loader,
    config=outer_config,
    device=device,
)
inner = Classifier(
    name="classifier",
    module=classifier_net,
    optimizer=classifier_optimizer,
    scheduler=classifier_scheduler,
    train_data_loader=classifier_loader,
    config=inner_config,
    device=device,
)

problems = [outer, inner]
l2u = {inner: [outer]}
u2l = {outer: [inner]}
dependencies = {"l2u": l2u, "u2l": u2l}

engine = NASEngine(config=engine_config, problems=problems, dependencies=dependencies)
engine.run()
