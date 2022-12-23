import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig
from betty.envs import Env

from support.omniglot_loader import OmniglotNShot


argparser = argparse.ArgumentParser()
argparser.add_argument("--n_way", type=int, help="n way", default=5)
argparser.add_argument("--k_spt", type=int, help="k shot for support set", default=5)
argparser.add_argument("--k_qry", type=int, help="k shot for query set", default=15)
argparser.add_argument(
    "--inner_steps", type=int, help="number of inner steps", default=5
)
argparser.add_argument(
    "--task_num", type=int, help="meta batch size, namely task num", default=16
)
argparser.add_argument("--seed", type=int, help="random seed", default=1)
arg = argparser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
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
)

db_test = OmniglotNShot(
    "/tmp/omniglot-data-test",
    batchsz=arg.task_num,
    n_way=arg.n_way,
    k_shot=arg.k_spt,
    k_query=arg.k_qry,
    imgsz=28,
    mode="test",
)


class Net(nn.Module):
    def __init__(self, n_way):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64, momentum=1, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64, n_way),
        )

    def forward(self, x):
        return self.net(x)


parent_module = Net(arg.n_way)
parent_optimizer = optim.Adam(parent_module.parameters(), lr=0.001)
parent_scheduler = optim.lr_scheduler.StepLR(parent_optimizer, step_size=100, gamma=0.9)


class Outer(ImplicitProblem):
    def training_step(self, batch):
        inputs, labels = batch
        inputs, labels = inputs.cuda(), labels.cuda()
        accs = []
        loss = 0
        for idx in range(len(self._children)):
            net = getattr(self, f"inner_{idx}")
            out = net(inputs[idx])
            loss += F.cross_entropy(out, labels[idx]) / len(self._children)
            accs.append((out.argmax(dim=1) == labels[idx]).detach())
        acc = 100.0 * torch.cat(accs).float().mean().item()

        return {"loss": loss, "acc": acc}

    def get_batch(self):
        x_spt, y_spt, x_qry, y_qry = self.env.batch
        return (x_qry, y_qry)


class Inner(ImplicitProblem):
    def training_step(self, batch):
        inputs, labels = batch
        inputs, labels = inputs.cuda(), labels.cuda()
        idx = self.outer.children.index(self)
        out = self.module(inputs[idx])
        loss = F.cross_entropy(out, labels[idx]) + self.reg_loss()

        return loss

    def reg_loss(self):
        return 0.5 * sum(
            [
                (p1 - p2).pow(2).sum()
                for p1, p2 in zip(
                    self.trainable_parameters(), self.outer.trainable_parameters()
                )
            ]
        )

    def get_batch(self):
        x_spt, y_spt, x_qry, y_qry = self.env.batch
        return (x_spt, y_spt)

    def on_inner_loop_start(self):
        self.module.load_state_dict(self.outer.module.state_dict())

    def configure_module(self):
        return Net(arg.n_way)

    def configure_optimizer(self):
        return optim.SGD(self.module.parameters(), lr=0.1)


class MAMLEnv(Env):
    def __init__(self):
        super().__init__()

        self.data_loader = db
        self.batch = None

    def step(self):
        try:
            self.batch = next(self.data_loader)
        except StopIteration:
            self.data_iterator = iter(db)
            self.batch = next(self.data_loader)


class MAMLEngine(Engine):
    def train_step(self):
        if self.global_step % arg.inner_steps == 1 or arg.inner_steps == 1:
            self.env.step()
        for leaf in self.leaves:
            leaf.step(global_step=self.global_step)

    def validation(self):
        if not hasattr(self, "best_acc"):
            self.best_acc = -1
        test_iter = len(db_test.datasets_cache["test"]) // arg.task_num + 1
        test_loader = iter(db_test)
        test_net = Net(arg.n_way).to(device)
        test_optim = optim.SGD(test_net.parameters(), lr=0.1)
        accs = []
        for _ in range(test_iter):
            batch = next(test_loader)
            x_spts, y_spts, x_qrys, y_qrys = batch
            x_spts, y_spts, x_qrys, y_qrys = (
                x_spts.to(device),
                y_spts.to(device),
                x_qrys.to(device),
                y_qrys.to(device),
            )
            for x_spt, y_spt, x_qry, y_qry in zip(x_spts, y_spts, x_qrys, y_qrys):
                test_net.load_state_dict(self.outer.module.state_dict())
                for _ in range(arg.inner_steps):
                    out = test_net(x_spt)
                    loss = F.cross_entropy(out, y_spt)
                    test_optim.zero_grad()
                    loss.backward()
                    test_optim.step()

                out = test_net(x_qry)
                accs.append((out.argmax(dim=1) == y_qry).detach())

        acc = 100.0 * torch.cat(accs).float().mean().item()
        if acc > self.best_acc:
            self.best_acc = acc

        return {"acc": acc, "best_acc": self.best_acc}


engine_config = EngineConfig(valid_step=100)
parent_config = Config(log_step=10, retain_graph=True)
child_config = Config(type="darts", unroll_steps=arg.inner_steps)

parent = Outer(
    name="outer",
    module=parent_module,
    optimizer=parent_optimizer,
    scheduler=parent_scheduler,
    config=parent_config,
    device=device,
)
children = [
    Inner(name=f"inner_{i}", config=child_config, device=device)
    for i in range(arg.task_num)
]
env = MAMLEnv()
problems = children + [parent]
u2l = {parent: children}
l2u = {}
for c in children:
    l2u[c] = [parent]
dependencies = {"u2l": u2l, "l2u": l2u}
engine = MAMLEngine(
    config=engine_config, problems=problems, dependencies=dependencies, env=env
)
engine.run()
