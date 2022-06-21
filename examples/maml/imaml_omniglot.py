import argparse
import sys

sys.path.insert(0, "./../..")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config

from support.omniglot_loader import OmniglotNShot


argparser = argparse.ArgumentParser()
argparser.add_argument("--n_way", type=int, help="n way", default=5)
argparser.add_argument("--k_spt", type=int, help="k shot for support set", default=5)
argparser.add_argument("--k_qry", type=int, help="k shot for query set", default=15)
argparser.add_argument("--inner_steps", type=int, help="number of inner steps", default=5)
argparser.add_argument("--task_num", type=int, help="meta batch size, namely task num", default=16)
argparser.add_argument("--seed", type=int, help="random seed", default=1)
arg = argparser.parse_args()

torch.manual_seed(arg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(arg.seed)
np.random.seed(arg.seed)

db = OmniglotNShot(
    "/tmp/omniglot-data",
    batchsz=arg.task_num,
    n_way=arg.n_way,
    k_shot=arg.k_spt,
    k_query=arg.k_qry,
    imgsz=28,
    device=arg.device,
)

db_test = OmniglotNShot(
    "/tmp/omniglot-data-test",
    batchsz=arg.task_num,
    n_way=arg.n_way,
    k_shot=arg.k_spt,
    k_query=arg.k_qry,
    imgsz=28,
    device=arg.device,
    mode="test",
)


class Net(nn.Module):
    def __init__(self, n_way):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64, n_way),
        )

    def forward(self, x):
        return self.net(x)


parent_module = Net(arg.n_way)
parent_optimizer = optim.Adam(parent_module.parameters(), lr=0.001)
parent_scheduler = optim.lr_scheduler.StepLR(parent_optimizer, step_size=50, gamma=0.9)


class Outer(ImplicitProblem):
    def training_step(self, batch):
        inputs, labels = batch
        accs = []
        loss = 0
        for idx in range(len(self._children)):
            net = getattr(self, f"inner_{idx}")
            out = net(inputs[idx])
            loss += F.cross_entropy(out, labels[idx]) / len(self._children)
            accs.append((out.argmax(dim=1) == labels[idx]).detach())
        acc = 100.0 * torch.cat(accs).float().mean().item()

        return {"loss": loss, "acc": acc}


class Inner(ImplicitProblem):
    def training_step(self, batch):
        idx = self.outer.children.index(self)
        inputs, labels = batch
        out = self.module(inputs[idx])
        loss = F.cross_entropy(out, labels[idx]) + self.reg_loss()

        return loss

    def reg_loss(self):
        return 0.25 * sum(
            [(p1 - p2).pow(2).sum() for p1, p2 in zip(self.parameters(), self.outer.parameters())]
        )

    def on_inner_loop_start(self):
        self.module.load_state_dict(self.outer.module.state_dict())


class MAMLEngine(Engine):
    def train_step(self):
        for leaf in self.leaves:
            leaf.step(global_step=self.global_step)


parent_config = Config(log_step=10)
child_config = Config(type="darts", unroll_steps=arg.inner_steps)

parent = Outer(
    name="outer",
    module=parent_module,
    optimizer=parent_optimizer,
    scheduler=parent_scheduler,
    config=parent_config)
children = []
for _ in range(arg.task_num):
    child_module = Net(arg.n_way)
    child_optimizer = optim.SGD(child_module.parameters(), lr=0.1)
    child = Inner(name="inner", module=child_module, optimizer=child_optimizer, config=child_config)
    children.append(child)

problems = children + [parent]
u2l = {parent: children}
l2u = {}
for c in children:
    l2u[c] = [parent]
dependencies = {"u2l": u2l, "l2u": l2u}
engine = Engine(config=None, problems=problems, dependencies=dependencies)
engine.run()
