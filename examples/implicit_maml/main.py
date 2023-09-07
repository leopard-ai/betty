import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import learn2learn as l2l

from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig
from betty.envs import Env

from models import ConvOmniglot, ConvMiniImagenet, ResNet12


argparser = argparse.ArgumentParser()
argparser.add_argument("--task", type=str, help="task", default="omniglot")
argparser.add_argument("--ways", type=int, help="n ways", default=5)
argparser.add_argument("--shots", type=int, help="k shots (train)", default=5)
argparser.add_argument("--inner_steps", type=int, help="num inner steps", default=5)
argparser.add_argument(
    "--meta_batch_size", type=int, help="meta batch size", default=16
)
argparser.add_argument(
    "--task_num", type=int, help="number of tasks for MAML", default=-1
)
argparser.add_argument("--seed", type=int, help="random seed", default=1)
argparser.add_argument("--hidden_size", type=int, default=64)
argparser.add_argument("--reg", type=float, default=0.5)
argparser.add_argument("--meta_lr", type=float, default=5e-4)
argparser.add_argument("--base_lr", type=float, default=1e-1)
argparser.add_argument("--model_type", type=str, default="cnn")
args = argparser.parse_args()

# Random seed setup
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# data loader
tasksets = l2l.vision.benchmarks.get_tasksets(
    args.task,
    train_ways=args.ways,
    train_samples=2 * args.shots,
    test_ways=args.ways,
    test_samples=2 * args.shots,
    num_tasks=args.task_num,
    root="./data",
)

# module & optimizer setup
model_cls = None
if args.task == "omniglot":
    model_cls = ConvOmniglot
elif args.task == "mini-imagenet":
    if args.model_type == "cnn":
        model_cls = ConvMiniImagenet
    elif args.model_type == "resnet":
        model_cls = ResNet12
    elif args.model_type == "wrn":
        model_cls = l2l.vision.models.WRN28

parent_module = (
    model_cls(args.ways)
    if args.model_type == "wrn"
    else model_cls(args.ways, args.hidden_size)
)
parent_optimizer = optim.AdamW(
    parent_module.parameters(), lr=args.meta_lr, weight_decay=1e-4
)
parent_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    parent_optimizer,
    T_max=int(args.meta_batch_size * 7500),
)

child_module = (
    model_cls(args.ways)
    if args.model_type == "wrn"
    else model_cls(args.ways, args.hidden_size)
)
child_optimizer = optim.SGD(child_module.parameters(), lr=args.base_lr)


def reg_loss(parameters, reference_parameters, reg_lambda=0.25):
    loss = 0
    for p1, p2 in zip(parameters, reference_parameters):
        loss += torch.sum(torch.pow((p1 - p2), 2))

    return reg_lambda * loss


def split_data(data, labels, shots, ways):
    out = {"train": None, "test": None}
    adapt_indices = np.zeros(data.size(0), dtype=bool)
    adapt_indices[np.arange(shots * ways) * 2] = True
    eval_indices = torch.from_numpy(~adapt_indices)
    adapt_indices = torch.from_numpy(adapt_indices)
    out["train"] = (data[adapt_indices], labels[adapt_indices])
    out["test"] = (data[eval_indices], labels[eval_indices])

    return out


class Outer(ImplicitProblem):
    def training_step(self, batch):
        inputs, labels = batch
        out = self.inner(inputs)
        loss = F.cross_entropy(out, labels)
        acc = 100.0 * (out.argmax(dim=1) == labels).float().mean().item()

        return {"loss": loss, "acc": acc}

    def get_batch(self):
        inputs, labels = self.env.batch["test"]

        return inputs, labels


class Inner(ImplicitProblem):
    def training_step(self, batch):
        inputs, labels = batch
        out = self.module(inputs)
        loss = F.cross_entropy(out, labels)
        reg = reg_loss(self.parameters(), self.outer.parameters(), args.reg)

        return loss + reg

    def get_batch(self):
        inputs, labels = self.env.batch["train"]

        return inputs, labels

    def on_inner_loop_start(self):
        self.module.load_state_dict(self.outer.module.state_dict())


class MAMLEnv(Env):
    def __init__(self):
        super().__init__()

        self.tasks = tasksets
        self.batch = {"train": None, "test": None}

    def step(self):
        data, labels = self.tasks.train.sample()
        data, labels = data.to(self.device), labels.to(self.device)
        out = split_data(data, labels, args.shots, args.ways)
        self.batch["train"] = out["train"]
        self.batch["test"] = out["test"]


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
        test_net = (
            model_cls(args.ways)
            if args.model_type == "wrn"
            else model_cls(args.ways, args.hidden_size)
        )
        test_optim = optim.SGD(test_net.parameters(), lr=args.base_lr)
        test_net = test_net.to(self.device)
        accs = []
        for i in range(500):
            inputs, labels = tasksets.test.sample()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            out = split_data(inputs, labels, args.shots, args.ways)
            train_inputs, train_labels = out["train"]
            test_inputs, test_labels = out["test"]
            test_net.load_state_dict(self.outer.module.state_dict())
            for _ in range(args.inner_steps):
                out = test_net(train_inputs)
                loss = F.cross_entropy(out, train_labels)
                test_optim.zero_grad()
                loss.backward()
                test_optim.step()

            out = test_net(test_inputs)
            accs.append((out.argmax(dim=1) == test_labels).detach())

        acc = 100.0 * torch.cat(accs).float().mean().item()
        if acc > self.best_acc:
            self.best_acc = acc

        return {"acc": acc, "best_acc": self.best_acc}


engine_config = EngineConfig(
    valid_step=int(args.inner_steps * args.meta_batch_size * 100),
    train_iters=int(args.inner_steps * args.meta_batch_size * 7500),
)
parent_config = Config(
    log_step=int(args.inner_steps * args.meta_batch_size * 10),
    retain_graph=True,
    gradient_accumulation=args.meta_batch_size,
)
# child_config = Config(type="darts", unroll_steps=args.inner_steps)
child_config = Config(
    type="cg", cg_iterations=3, cg_alpha=1.0, unroll_steps=args.inner_steps
)

outer = Outer(
    name="outer",
    module=parent_module,
    optimizer=parent_optimizer,
    scheduler=parent_scheduler,
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
