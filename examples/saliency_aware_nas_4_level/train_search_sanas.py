import os, time, glob
import logging
import argparse
import numpy as np

# import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem

# from model_search import Network, Architecture
# from model_search_pcdarts import Network, Architecture
import utils
from resnet import *
import copy
import sys

import math
import unittest


parser = argparse.ArgumentParser("cifar")
parser.add_argument(
    "--data", type=str, default="../data", help="location of the data corpus"
)
# parser.add_argument("--batchsz", type=int, default=64, help="batch size")
parser.add_argument("--batchsz", type=int, default=192, help="batch size")
parser.add_argument(
    "--warmup", type=int, default=10, help="num of training warmup epochs"
)
parser.add_argument(
    "--darts_type", type=str, default="PCDARTS", help="[DARTS, PCDARTS]"
)
parser.add_argument(
    "--dataset", type=str, default="cifar100", help="[cifar10, cifar100]"
)
# parser.add_argument("--lr", type=float, default=0.025, help="init learning rate")
# parser.add_argument("--lr_min", type=float, default=0.001, help="min learning rate")
parser.add_argument("--lr", type=float, default=0.1, help="init learning rate")
parser.add_argument("--lr_min", type=float, default=0.0, help="min learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--wd", type=float, default=3e-4, help="weight decay")
parser.add_argument("--report_freq", type=int, default=100, help="report frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument("--epochs", type=int, default=50, help="num of training epochs")
parser.add_argument("--init_ch", type=int, default=16, help="num of init channels")
parser.add_argument("--layers", type=int, default=8, help="total number of layers")
parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
parser.add_argument("--cutout_len", type=int, default=16, help="cutout length")
parser.add_argument("--save", type=str, default="EXP", help="experiment name")
parser.add_argument(
    "--drop_path_prob", type=float, default=0.3, help="drop path probability"
)
parser.add_argument(
    "--train_portion", type=float, default=0.5, help="portion of training/val splitting"
)
# parser.add_argument("--arch_lr", type=float, default=3e-4, help="learning rate for arch encoding")
# parser.add_argument("--arch_wd", type=float, default=1e-3, help="weight decay for arch encoding")
parser.add_argument(
    "--arch_lr", type=float, default=6e-4, help="learning rate for arch encoding"
)
parser.add_argument(
    "--arch_wd", type=float, default=1e-3, help="weight decay for arch encoding"
)
parser.add_argument("--arch_steps", type=int, default=4, help="architecture steps")
parser.add_argument("--unroll_steps", type=int, default=1, help="unrolling steps")
parser.add_argument("--lam", type=float, help="lambda", default=1)
parser.add_argument("--gamma", type=float, help="gamma", default=1)
args = parser.parse_args()

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(args.save, "log_lease.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("gpu device = %d" % args.gpu)
logging.info("args = %s", args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda:0")

if args.dataset == "cifar10":
    CIFAR_CLASS = 10
    train_transform, valid_transform = utils.data_transforms_cifar10(args)
    train_data = dset.CIFAR10(
        root=args.data, train=True, download=True, transform=train_transform
    )
    valid_data = dset.CIFAR10(
        root=args.data, train=False, download=True, transform=valid_transform
    )
elif args.dataset == "cifar100":
    CIFAR_CLASS = 100
    train_transform, valid_transform = utils.data_transforms_cifar100(args)
    train_data = dset.CIFAR100(
        root=args.data, train=True, download=True, transform=train_transform
    )
    valid_data = dset.CIFAR100(
        root=args.data, train=False, download=True, transform=valid_transform
    )

test_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batchsz, shuffle=False, pin_memory=True, num_workers=2
)

num_train = len(train_data)  # 50000
indices = list(range(num_train))
split = int(np.floor(args.train_portion * num_train))
warmup = args.warmup

if args.darts_type == "DARTS":
    from model_search import Network, Architecture
elif args.darts_type == "PCDARTS":
    from model_search_pcdarts import Network, Architecture

report_freq = int(num_train * args.train_portion // args.batchsz + 1)

train_iters = int(
    args.epochs
    * (num_train * args.train_portion // args.batchsz + 1)
    * args.unroll_steps
)


rand = True
num_steps = 7
epsilon = 8 / 255.0
step_size = 2 / 255.0


class Outer(ImplicitProblem):
    def forward(self):
        return self.module()

    def training_step(self, batch):
        x, target = batch
        x, target = x.to(device), target.to(device, non_blocking=True)

        alphas = self.forward()
        loss1 = self.inner1.module.loss(x, alphas, target)
        loss2 = self.inner2.module.loss(x, alphas, target)

        # for (n1, p1), (n2, p2) in zip(self.inner1.module.named_parameters(),self.inner2.module.named_parameters()):
        #     print(n1,n2)

        loss = loss2 + args.lam * loss1
        assert not math.isnan(loss)
        # epoch = int(int(self.count)//(num_train * args.train_portion // args.batchsz))
        # epoch = epoch // args.unroll_steps
        # epoch = int(int(self.count)//(num_train * args.train_portion // args.batchsz))
        # epoch = epoch // args.unroll_steps
        epoch = int(
            self.count
            * (args.batchsz + 1)
            * args.unroll_steps
            // (num_train * args.train_portion)
        )
        print(f"Epoch: {epoch} || step: {self.count} || loss: {loss.item()}")
        # if self.count % 50 == 0:
        #     print(f"step {self.count} || loss: {loss.item()}")

        return loss

    def configure_train_data_loader(self):
        valid_queue = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batchsz,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
            pin_memory=True,
            num_workers=2,
        )
        return valid_queue

    def configure_module(self):
        return Architecture(steps=args.arch_steps).to(device)

    def configure_optimizer(self):
        optimizer = optim.Adam(
            self.module.parameters(),
            lr=args.arch_lr,
            betas=(0.5, 0.999),
            weight_decay=args.arch_wd,
        )
        return optimizer


class Inner2(ImplicitProblem):
    def forward(self, pert_inp, alphas):
        return self.module(pert_inp, alphas)

    def training_step(self, batch):
        x, target = batch
        x, target = x.to(device), target.to(device, non_blocking=True)

        alphas = self.outer()

        # self.module = copy.deepcopy(self.inner1.module)
        # model = self.inner1.module

        pert_inp = self.attack(alphas, x, target)
        ##############################################################################
        # w2 = self.forward(pert_inp,alphas)
        ##############################################################################
        # loss = self.inner1.module.loss(pert_inp, alphas,target)

        # loss1 = self.inner1.module.loss(x, alphas, target)
        loss1 = self.inner1.module.loss(pert_inp, alphas, target)
        loss2 = self.module.loss(pert_inp, alphas, target)
        loss = loss2 + args.gamma * loss1

        # loss = self.module.loss(pert_inp, alphas,target)

        return loss

    def attack(self, alphas, x, target):
        # cost = nn.CrossEntropyLoss()
        # alphas = self.outer()
        x_purt = x.clone().detach()
        target = target.clone().detach()
        if rand:
            x_purt = x_purt + torch.zeros_like(x_purt).uniform_(-epsilon, epsilon)
        for i in range(num_steps):
            x_purt.requires_grad_()
            with torch.enable_grad():
                # ##############################################################################
                # # logits = self.inner1(x_purt, alphas)
                logits = self.inner1.module(x_purt, alphas)
                # ##############################################################################
                loss1 = F.cross_entropy(logits, target, reduction="none")
                loss1 = torch.mean(loss1)
                # loss1 = self.inner1.module.loss(x_purt, alphas,target)
            # x_purt.requires_grad = True
            # logits = model_attack(x_purt, alphas)
            # loss1 = cost(logits, target)
            # print(loss1)
            grad = torch.autograd.grad(loss1, [x_purt])[0]

            x_purt = x_purt.detach() + step_size * torch.sign(grad.detach())
            delta = torch.clamp(x_purt - x, min=-epsilon, max=epsilon)
            # self.delta = nn.Parameter(torch.clamp(x_purt - x, min=-epsilon, max=epsilon), requires_grad=True).to(device)
            x_purt = torch.clamp(x + delta, min=0, max=1).detach()

        # pert_inp = torch.mul(x, torch.round(torch.abs(delta) * 255/8 + 0.499))
        # pert_inp = torch.mul(x, torch.abs(delta)+1)
        pert_inp = torch.mul(x, delta)
        # pert_inp = torch.mul(x, torch.abs(delta))

        return pert_inp

    def configure_train_data_loader(self):
        train_queue = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batchsz,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True,
            num_workers=2,
        )
        return train_queue

    # def configure_module(self):
    #     criterion_audience = nn.CrossEntropyLoss().to(device)
    #     return ResNet(criterion_audience).to(device)
    def configure_module(self):
        criterion = nn.CrossEntropyLoss().to(device)
        return Network(
            args.init_ch, CIFAR_CLASS, args.layers, criterion, steps=args.arch_steps
        ).to(device)

    def configure_optimizer(self):
        optimizer = optim.SGD(
            self.module.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
        return optimizer

    def configure_scheduler(self):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, float(train_iters // args.unroll_steps), eta_min=args.lr_min
        )
        return scheduler


class Inner1(ImplicitProblem):
    def forward(self, x, alphas):
        return self.module(x, alphas)

    def training_step(self, batch):
        x, target = batch
        x, target = x.to(device), target.to(device, non_blocking=True)
        alphas = self.outer()
        loss = self.module.loss(x, alphas, target)
        return loss

    def configure_train_data_loader(self):
        train_queue = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batchsz,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True,
            num_workers=2,
        )
        return train_queue

    def configure_module(self):
        criterion = nn.CrossEntropyLoss().to(device)
        return Network(
            args.init_ch, CIFAR_CLASS, args.layers, criterion, steps=args.arch_steps
        ).to(device)

    def configure_optimizer(self):
        optimizer = optim.SGD(
            self.module.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
        return optimizer

    def configure_scheduler(self):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, float(train_iters // args.unroll_steps), eta_min=args.lr_min
        )
        return scheduler


class NASEngine(Engine):
    @torch.no_grad()
    def validation(self):
        corrects = 0
        total = 0
        for x, target in test_queue:
            x, target = x.to(device), target.to(device, non_blocking=True)
            alphas = self.outer()
            _, correct = self.inner1.module.loss(x, alphas, target, acc=True)
            corrects += correct
            total += x.size(0)
        acc = corrects / total

        logging.info("[*] Valid Acc.: %f", acc)
        print("[*] Valid Acc.:", acc)
        alphas = self.outer()
        logging.info("genotype = %s", self.inner1.module.genotype(alphas))
        torch.save({"genotype": self.inner1.module.genotype(alphas)}, "genotype.t7")


criterion = nn.CrossEntropyLoss().to(device)
# model_pert = Network(args.init_ch, 10, args.layers, criterion, steps=args.arch_steps).to(device)

# outer_config = Config(retain_graph=True, first_order=True,log_step=1, fp16=True)
# inner2_config = Config(type="darts", unroll_steps=args.unroll_steps, allow_unused=True, fp16=True)
# inner1_config = Config(type="darts", unroll_steps=args.unroll_steps, allow_unused=True, fp16=True)

outer_config = Config(retain_graph=True, first_order=True, log_step=1)
inner2_config = Config(type="darts", unroll_steps=args.unroll_steps, allow_unused=True)
inner1_config = Config(type="darts", unroll_steps=args.unroll_steps, allow_unused=True)
engine_config = EngineConfig(
    valid_step=report_freq,
    train_iters=train_iters,
    roll_back=True,
)
outer = Outer(name="outer", config=outer_config, device=device)
inner1 = Inner1(name="inner1", config=inner1_config, device=device)
inner2 = Inner2(name="inner2", config=inner2_config, device=device)


problems = [outer, inner2, inner1]
# l2u = {inner1: [inner2], inner2: [outer], inner1:[outer]}
# u2l = {outer: [inner1],outer:[inner2]}

# l2u = {inner1: [inner2,outer], inner1: [outer], inner2: [outer]}
# l2u = {inner1: [inner2,outer],  inner2: [outer]}
# u2l = {outer: [inner1,inner2]}
# u2l = {outer: [inner2,inner1],outer: [inner1]}

# l2u = {inner1: [outer], inner1: [inner2],  inner2: [outer], inner1: [inner2,outer]}
l2u = {inner1: [inner2, outer], inner2: [outer]}
u2l = {outer: [inner2, inner1]}

# problems = [outer, inner1]
# l2u = {inner1: [outer]}
# u2l = {outer: [inner1]}
# problems = [outer, inner2, perturb, inner1]
# l2u = {inner1: [inner2, perturb,outer], perturb: [inner2, outer], inner2: [outer]}
# u2l = {outer: [inner2, perturb,inner1]}
# u2l = {outer: [inner1], inner2: [inner1]}
# u2l = {outer: [inner1]}
# l2u = {inner1: [AttackPGD, inner2, outer], AttackPGD:[inner2, outer], inner2: [outer]}
# u2l = {outer: [inner2,AttackPGD, inner1]}
# u2l = {outer: [inner1], outer: [inner2, inner1]}
# u2l = {outer: [inner2], inner2:[inner1], outer: [inner1]}
# u2l = {outer: [inner2, inner1], inner2: [inner1]}
dependencies = {"l2u": l2u, "u2l": u2l}

engine = NASEngine(config=engine_config, problems=problems, dependencies=dependencies)
engine.run()
