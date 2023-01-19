import os, time, glob
import logging
import argparse
import numpy as np
import math

import torch
import torch.nn as nn
from torch import optim
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem

# from model_search import Network, Architecture
# from model_search_pcdarts import Network, Architecture

import utils
from regularizer import *
import sys


parser = argparse.ArgumentParser("cifar")
parser.add_argument(
    "--data", type=str, default="../data", help="location of the data corpus"
)
parser.add_argument("--batchsz", type=int, default=64, help="batch size")
# parser.add_argument("--batchsz", type=int, default=128, help="batch size")
parser.add_argument("--lambda", type=float, default=0.01, help="tradeoff param")
parser.add_argument("--lr", type=float, default=0.025, help="init learning rate")
parser.add_argument("--lr_min", type=float, default=0.001, help="min learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--wd", type=float, default=3e-4, help="weight decay")
parser.add_argument("--report_freq", type=int, default=100, help="report frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument("--epochs", type=int, default=50, help="num of training epochs")
parser.add_argument(
    "--warmup", type=int, default=10, help="num of training warmup epochs"
)
parser.add_argument("--init_ch", type=int, default=16, help="num of init channels")
parser.add_argument("--layers", type=int, default=8, help="total number of layers")
parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
parser.add_argument("--cutout_len", type=int, default=16, help="cutout length")
parser.add_argument("--save", type=str, default="EXP", help="experiment name")
parser.add_argument("--darts_type", type=str, default="DARTS", help="[DARTS, PCDARTS]")
parser.add_argument(
    "--loss_type",
    type=str,
    default="loss_hessian",
    help="type of loss [loss_hessian,jacob]",
)
parser.add_argument(
    "--training",
    type=str,
    default="regulizer",
    help="type of training: [standard, regulizer]",
)
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

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda:0")

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

report_freq = int(num_train * args.train_portion // args.batchsz + 1)

train_iters = int(args.epochs * report_freq * args.unroll_steps)

# print(train_iters)
lambda_JR = 0.01
lambda_JR2 = 1e-4
loss_type = str(args.loss_type)
training = str(args.training)
warmup = args.warmup

if args.darts_type == "DARTS":
    from model_search import Network, Architecture
elif args.darts_type == "PCDARTS":
    from model_search_pcdarts import Network, Architecture

h_all = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5])
h_all = np.append(h_all, [1.5] * int(args.epochs - 6))


class Outer(ImplicitProblem):
    def forward(self):
        return self.module()

    def training_step(self, batch):
        x, target = batch
        x, target = x.to(device), target.to(device, non_blocking=True)
        alphas = self.forward()
        # epoch = int(int(self.count)//(num_train * args.train_portion // args.batchsz))
        # epoch = epoch // args.unroll_steps
        epoch = int(
            self.count
            * (args.batchsz + 1)
            * args.unroll_steps
            // (num_train * args.train_portion)
        )
        h = h_all[epoch]
        if epoch < warmup:
            loss, acc = self.inner.module.loss(x, alphas, target, acc=True)
            print(
                f"Epoch: {epoch} || step: {self.count} || loss: {loss.item()} || acc: {acc/args.unroll_steps}"
            )
        else:
            loss = self.total_loss(batch, alphas, lambda_JR, h)
            print(f"Epoch: {epoch} || step: {self.count} || loss: {loss.item()}")
        # if self.count % 50 == 0:
        #     print(f"step {self.count} || loss: {loss.item()}")

        return loss

    def total_loss(self, batch, alphas, lambda_JR, h):
        x, target = batch
        x, target = x.to(device), target.to(device, non_blocking=True)
        x.requires_grad = True
        loss_super = self.inner.module.loss(x, alphas, target)
        if loss_type == "loss_hessian":
            criterion = nn.CrossEntropyLoss().to(device)
            reg = loss_curv(self.inner.module, criterion, lambda_=4, device="cuda")
            regularizer, grad_norm = reg.regularizer(x, alphas, target, h=h)
        elif loss_type == "jacob":
            logits = self.inner.module(x, alphas)
            n_proj = 1
            # reg = JacobianReg(n=n_proj)
            reg = JacobiNormReg(n=n_proj)
            # reg = PJacobiNormReg(n=n_proj)
            regularizer = reg(x, logits)
        # del reg
        # del logits
        # torch.cuda.empty_cache()
        # x.requires_grad = False

        # return loss_super - lambda_JR * loss_JR
        return loss_super + lambda_JR * regularizer

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


class Inner(ImplicitProblem):
    def forward(self, x, alphas):
        return self.module(x, alphas)

    def training_step(self, batch):
        x, target = batch
        x, target = x.to(device), target.to(device, non_blocking=True)
        alphas = self.outer()
        if training == "standard":
            loss = self.module.loss(x, alphas, target)
        else:
            # epoch = int(int(self.count)//(num_train * args.train_portion // args.batchsz))
            # epoch = epoch // args.unroll_steps
            epoch = int(
                self.count
                * (args.batchsz + 1)
                * args.unroll_steps
                // (num_train * args.train_portion)
            )

            h = h_all[epoch]
            if epoch < warmup:
                loss, acc = self.module.loss(x, alphas, target, acc=True)
            else:
                loss = self.total_loss(batch, alphas, lambda_JR2, h)

        return loss

    def total_loss(self, batch, alphas, lambda_JR, h):
        x, target = batch
        x, target = x.to(device), target.to(device, non_blocking=True)
        x.requires_grad = True
        loss_super = self.module.loss(x, alphas, target)
        if loss_type == "loss_hessian":
            criterion = nn.CrossEntropyLoss().to(device)
            reg = loss_curv(self.module, criterion, lambda_=1, device="cuda")
            regularizer, grad_norm = reg.regularizer(x, alphas, target, h=h)
        elif loss_type == "jacob":
            logits = self.module(x, alphas)
            n_proj = 1
            # reg = JacobianReg(n=n_proj)
            reg = JacobiNormReg(n=n_proj)
            # reg = PJacobiNormReg(n=n_proj)
            regularizer = reg(x, logits)
        # del reg
        # del logits
        # torch.cuda.empty_cache()
        # x.requires_grad = False

        # return loss_super - lambda_JR * loss_JR
        return loss_super + lambda_JR * regularizer

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
            args.init_ch, 10, args.layers, criterion, steps=args.arch_steps
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
            _, correct = self.inner.module.loss(x, alphas, target, acc=True)
            corrects += correct
            total += x.size(0)
        acc = corrects / total

        logging.info("[*] Valid Acc.: %f", acc)
        print("[*] Valid Acc.:", acc)
        alphas = self.outer()
        logging.info("genotype = %s", self.inner.module.genotype(alphas))
        torch.save({"genotype": self.inner.module.genotype(alphas)}, "genotype.t7")


# outer_config = Config(retain_graph=True, first_order=True,log_step=1, fp16=True)
# inner_config = Config(type="darts", unroll_steps=args.unroll_steps, fp16=True)
outer_config = Config(retain_graph=True, first_order=True, log_step=1)
inner_config = Config(type="darts", unroll_steps=args.unroll_steps)
engine_config = EngineConfig(
    valid_step=report_freq,
    train_iters=train_iters,
    roll_back=True,
)
outer = Outer(name="outer", config=outer_config, device=device)
inner = Inner(name="inner", config=inner_config, device=device)

problems = [outer, inner]
l2u = {inner: [outer]}
u2l = {outer: [inner]}
dependencies = {"l2u": l2u, "u2l": u2l}

engine = NASEngine(config=engine_config, problems=problems, dependencies=dependencies)
engine.run()
