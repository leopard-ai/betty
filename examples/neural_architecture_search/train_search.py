import sys
sys.path.insert(0, "./../..")

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
from betty.config_template import Config
from betty.problems import ImplicitProblem

from model_search import Network, Architecture
import utils


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batchsz', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.025, help='init learning rate')
parser.add_argument('--lr_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--wd', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_ch', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_len', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--exp_path', type=str, default='search', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping range')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training/val splitting')
parser.add_argument('--arch_lr', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_wd', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--arch_steps', type=int, default=4, help='architecture steps')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda:0')

train_transform, valid_transform = utils.data_transforms_cifar10(args)
train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

num_train = len(train_data) # 50000
indices = list(range(num_train))
split = int(np.floor(args.train_portion * num_train))


class Outer(ImplicitProblem):
    def forward(self):
        return self.module()

    def training_step(self, batch):
        x, target = batch
        x, target = x.to(device), target.to(device, non_blocking=True)

        alphas = self.forward()
        loss = self.inner.module.loss(x, alphas, target)

        if self.count % 10 == 0:
            print(f"step {self.count} || loss: {loss.item()}")

        return loss

    def configure_train_data_loader(self):
        valid_queue = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batchsz,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
            pin_memory=True,
            num_workers=2
        )
        return valid_queue

    def configure_module(self):
        return Architecture(steps=args.arch_steps).to(device)

    def configure_optimizer(self):
        optimizer = optim.Adam(self.module.parameters(),
                               lr=args.arch_lr,
                               betas=(0.5, 0.999),
                               weight_decay=args.arch_wd)
        return optimizer


class Inner(ImplicitProblem):
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
            num_workers=2
        )
        return train_queue

    def configure_module(self):
        criterion = nn.CrossEntropyLoss().to(device)
        return Network(args.init_ch, 10, args.layers, criterion, steps=args.arch_steps).to(device)

    def configure_optimizer(self):
        optimizer = optim.SGD(self.module.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.wd)
        return optimizer

    def configure_scheduler(self):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                         float(args.epochs),
                                                         eta_min=args.lr_min)
        return scheduler

outer_config = Config(type='darts',
                      step=1,
                      retain_graph=True,
                      first_order=True)
inner_config = Config(type='torch',
                      step=1,
                      first_order=True,
                      retain_graph=True)
outer = Outer(name='outer', config=outer_config, device=device)
inner = Inner(name='inner', config=inner_config, device=device)

problems = [outer, inner]
dependencies = {outer: [inner]}

engine = Engine(config=None, problems=problems, dependencies=dependencies)
engine.run()
