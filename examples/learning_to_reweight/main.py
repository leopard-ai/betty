import sys
sys.path.insert(0, "./../..")
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import *
from data import *
from utils import *

from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.config_template import Config


parser = argparse.ArgumentParser(description='Meta_Weight_Net')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--meta_net_hidden_size', type=int, default=100)
parser.add_argument('--meta_net_num_layers', type=int, default=1)

parser.add_argument('--lr', type=float, default=.1)
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--dampening', type=float, default=0.)
parser.add_argument('--nesterov', type=bool, default=False)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--meta_lr', type=float, default=1e-5)
parser.add_argument('--meta_weight_decay', type=float, default=0.)

parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--imbalanced_factor', type=int, default=None)
parser.add_argument('--corruption_type', type=str, default=None)
parser.add_argument('--corruption_ratio', type=float, default=0.)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--max_epoch', type=int, default=120)

parser.add_argument('--meta_interval', type=int, default=1)
parser.add_argument('--paint_interval', type=int, default=20)

args = parser.parse_args()
print(args)

train_dataloader, meta_dataloader, test_dataloader, imbalanced_num_list = build_dataloader(
    seed=args.seed,
    dataset=args.dataset,
    num_meta_total=args.num_meta,
    imbalanced_factor=args.imbalanced_factor,
    corruption_type=args.corruption_type,
    corruption_ratio=args.corruption_ratio,
    batch_size=args.batch_size
)

class Outer(ImplicitProblem):
    def forward(self, x):
        return self.module(x)

    def training_step(self, batch):
        inputs, labels = batch
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = self.inner(inputs)
        loss = F.cross_entropy(outputs, labels.long())

        if self.count % 10 == 0:
            acc = (outputs.argmax(dim=1) == labels.long()).float().mean().item() * 100
            print(f"step {self.count} || acc: {acc}")

        return loss

    def configure_train_data_loader(self):
        return meta_dataloader

    def configure_module(self):
        meta_net = MLP(hidden_size=args.meta_net_hidden_size,
                       num_layers=args.meta_net_num_layers).to(device=args.device)
        return meta_net

    def configure_optimizer(self):
        meta_optimizer = optim.Adam(self.module.parameters(),
                                    lr=args.meta_lr,
                                    weight_decay=args.meta_weight_decay)
        return meta_optimizer


class Inner(ImplicitProblem):
    def forward(self, x):
        return self.module(x)

    def training_step(self, batch):
        inputs, labels = batch
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        outputs = self.forward(inputs)
        loss_vector = F.cross_entropy(outputs, labels.long(), reduction='none')
        loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))
        weight = self.outer(loss_vector_reshape.detach())
        loss = torch.mean(weight * loss_vector_reshape)
        self.scheduler.step()

        return loss

    def configure_train_data_loader(self):
        return train_dataloader

    def configure_module(self):
        return ResNet32(args.dataset == 'cifar10' and 10 or 100).to(device=args.device)

    def configure_optimizer(self):
        optimizer = optim.SGD(self.module.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              dampening=args.dampening,
                              weight_decay=args.weight_decay,
                              nesterov=args.nesterov)
        return optimizer

    def configure_scheduler(self):
        scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                   milestones=[7500, 12000],
                                                   gamma=0.1)
        return scheduler

outer_config = Config(type='darts',
                      step=5,
                      retain_graph=True,
                      first_order=True)
inner_config = Config(type='torch')
outer = Outer(name='outer', config=outer_config, device=args.device)
inner = Inner(name='inner', config=inner_config, device=args.device)

problems = [outer, inner]
h2l = {outer: [inner]}
l2h = {inner: [outer]}
dependencies = {'l2h': l2h, 'h2l': h2l}

engine = Engine(config=None, problems=problems, dependencies=dependencies)
engine.run()
