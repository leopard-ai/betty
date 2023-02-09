import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig
from betty.envs import Env

from models import OmniglotCNN, MiniImagenetCNN


argparser = argparse.ArgumentParser()
argparser.add_argument("--task", type=str, help="task", default="omniglot")
argparser.add_argument("--ways", type=int, help="n ways", default=5)
argparser.add_argument("--shots", type=int, help="k shots (train)", default=5)
argparser.add_argument("--test_shots", type=int, help="k shots (test)", default=15)
argparser.add_argument("--inner_steps", type=int, help="num inner steps", default=5)
argparser.add_argument("--task_num", type=int, help="meta batch size", default=16)
argparser.add_argument("--seed", type=int, help="random seed", default=1)
argparser.add_argument('--n_conv', type=int, default=4)
argparser.add_argument('--n_dense', type=int, default=0)
argparser.add_argument('--hidden_dim', type=int, default=64)
argparser.add_argument('--in_channels', type=int, default=1)
argparser.add_argument('--hidden_channels', type=int, default=64)
args = argparser.parse_args()

# Random seed setup
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# data loader
dataset_cls = None
if args.task == "omniglot":
    dataset_cls = omniglot
elif args.task == "miniimagenet":
    dataset_cls = miniimagenet

train_set = dataset_cls(
    "data",
    ways=args.ways,
    shots=args.shots,
    test_shots=args.test_shots,
    meta_train=True,
    download=True,
)
train_loader = BatchMetaDataLoader(
    train_set,
    batch_size=1,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
)
test_set = dataset_cls(
    "data",
    ways=args.ways,
    shots=args.shots,
    test_shots=args.test_shots,
    meta_val=True,
    download=True,
)
test_loader = BatchMetaDataLoader(
    test_set,
    batch_size=1,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
)


# module & optimizer setup
model_cls = None
if args.task == "omniglot":
    model_cls = OmniglotCNN
elif args.task == "miniimagenet":
    model_cls = MiniImagenetCNN
parent_module = model_cls(args.ways)
parent_optimizer = optim.Adam(parent_module.parameters(), lr=3e-4)

child_module = model_cls(args.ways)
child_optimizer = optim.SGD(child_module.parameters(), lr=1e-1)


def reg_loss(parameters, reference_parameters, reg_lambda=0.5):
    loss = 0
    for p1, p2 in zip(parameters, reference_parameters):
        loss += torch.sum(torch.pow((p1 - p2), 2))

    return reg_lambda * loss


class Outer(ImplicitProblem):
    def training_step(self, batch):
        inputs, labels = batch
        out = self.inner(inputs)
        loss = F.cross_entropy(out, labels)
        acc = 100.0 * (out.argmax(dim=1) == labels).float().mean().item()

        return {"loss": loss, "acc": acc}

    def get_batch(self):
        inputs, labels = self.env.batch["test"]

        return inputs[0].to(self.device), labels[0].to(self.device)


class Inner(ImplicitProblem):
    def training_step(self, batch):
        inputs, labels = batch
        out = self.module(inputs)
        loss = F.cross_entropy(out, labels)
        reg = reg_loss(self.parameters(), self.outer.parameters())

        return loss + reg

    def get_batch(self):
        inputs, labels = self.env.batch["train"]

        return inputs[0].to(self.device), labels[0].to(self.device)

    def on_inner_loop_start(self):
        self.module.load_state_dict(self.outer.module.state_dict())


class MAMLEnv(Env):
    def __init__(self):
        super().__init__()

        self.train_data_loader = train_loader
        self.train_data_iterator = iter(self.train_data_loader)
        self.batch = None

    def step(self):
        try:
            self.batch = next(self.train_data_iterator)
        except StopIteration:
            self.train_data_iterator = iter(self.train_data_loader)
            self.batch = next(self.train_data_iterator)


class MAMLEngine(Engine):
    def train_step(self):
        if self.global_step % args.inner_steps == 1 or args.inner_steps == 1:
            self.env.step()
        for leaf in self.leaves:
            leaf.step(global_step=self.global_step)

    def validation(self):
        self.outer.module.train()
        if not hasattr(self, "best_acc"):
            self.best_acc = -1
        test_net = model_cls(args.ways).to(self.device)
        test_optim = optim.SGD(test_net.parameters(), lr=0.1)
        accs = []
        for idx, batch in enumerate(test_loader):
            if idx == 500:
                break
            train_inputs, train_labels = batch["train"]
            test_inputs, test_labels = batch["test"]
            train_inputs = train_inputs[0].to(self.device)
            train_labels = train_labels[0].to(self.device)
            test_inputs = test_inputs[0].to(self.device)
            test_labels = test_labels[0].to(self.device)
            test_net.load_state_dict(self.outer.module.state_dict())
            for _ in range(args.inner_steps):
                out = test_net(train_inputs)
                loss = F.cross_entropy(out, train_labels) + reg_loss(
                    list(test_net.parameters()), self.outer.parameters()
                )
                #loss = F.cross_entropy(out, train_labels)
                test_optim.zero_grad()
                loss.backward()
                test_optim.step()

            out = test_net(test_inputs)
            accs.append((out.argmax(dim=1) == test_labels).detach())

        acc = 100.0 * torch.cat(accs).float().mean().item()
        if acc > self.best_acc:
            self.best_acc = acc

        return {"acc": acc, "best_acc": self.best_acc}


engine_config = EngineConfig(valid_step=20000, train_iters=1000000)
parent_config = Config(
    log_step=800, retain_graph=True, gradient_accumulation=args.task_num
)
child_config = Config(type="darts", unroll_steps=args.inner_steps)

outer = Outer(
    name="outer",
    module=parent_module,
    optimizer=parent_optimizer,
    config=parent_config,
)
inner = Inner(
    name="inner", module=child_module, optimizer=child_optimizer, config=child_config
)
env = MAMLEnv()
problems = [outer, inner]
u2l = {outer: [inner]}
l2u = {inner: [outer]}
dependencies = {"u2l": u2l, "l2u": l2u}
engine = MAMLEngine(
    config=engine_config, problems=problems, dependencies=dependencies, env=env
)
engine.run()
