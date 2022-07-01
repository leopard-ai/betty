import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig

from utils.utils import set_random_seed
from utils.data_transform import transform
from dalib.vision.datasets import Office31, OfficeHome
from model.resnet import build_model, build_optimizer, MLP


domainIdxDict = {"Ar": 0, "Cl": 1, "Pr": 2, "Rw": 3, "A": 0, "D": 1, "W": 2}


def argument_parser():
    parser = argparse.ArgumentParser(description="regularize the target by the source")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--source_domain", type=str, default="Cl")
    parser.add_argument("--target_domain", type=str, default="Ar")
    parser.add_argument("--features_lr", type=float, default=1e-4)
    parser.add_argument("--classifier_lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="officehome")
    parser.add_argument("--data_dir", type=str, metavar="PATH", default="./data")
    parser.add_argument("--lam", type=float, help="lambda", default=7e-3)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--step_size", type=int, default=400)
    parser.add_argument("--train_portion", type=float, default=0.9)
    parser.add_argument("--baseline", action="store_true", default=False)

    return parser


parser = argument_parser()
args = parser.parse_args()

set_random_seed(args.random_seed)
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# Dataset
train_transform = transform(train=True)
test_transform = transform(train=False)
data_root = os.path.join(args.data_dir, args.dataset)
if args.dataset == "office31":
    getDataset = Office31
elif args.dataset == "officehome":
    getDataset = OfficeHome

print(f"train source_task: {args.source_domain}")
train_source_dataset = getDataset(
    root=data_root,
    task=args.source_domain + "_train",
    download=True,
    transform=train_transform,
)

print(f"train target_task: {args.target_domain}")
train_target_dataset = getDataset(
    root=data_root,
    task=args.target_domain + "_train",
    download=True,
    transform=train_transform,
)
valid_target_dataset = getDataset(
    root=data_root,
    task=args.target_domain + "_train",
    download=True,
    transform=test_transform,
)

test_target_dataset = getDataset(
    root=data_root,
    task=args.target_domain + "_test",
    download=True,
    transform=test_transform,
)

train_loader = torch.utils.data.DataLoader(
    train_target_dataset,
    batch_size=args.batch_size,
    num_workers=6,
    shuffle=True,
    pin_memory=True,
    drop_last=False,
)
valid_loader = torch.utils.data.DataLoader(
    valid_target_dataset,
    batch_size=args.batch_size,
    num_workers=6,
    shuffle=True,
    pin_memory=True,
    drop_last=False,
)
test_loader = torch.utils.data.DataLoader(
    test_target_dataset,
    batch_size=args.batch_size,
    num_workers=6,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)

train_source_loader = torch.utils.data.DataLoader(
    train_source_dataset,
    batch_size=args.batch_size,
    num_workers=6,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
)


class ReweightingModule(torch.nn.Module):
    def __init__(self, dim):
        super(ReweightingModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.zeros(dim))

    def forward(self):
        return self.weight


class Pretraining(ImplicitProblem):
    def forward(self, x):
        return self.module(x)

    def training_step(self, batch):
        inputs, targets, _ = batch
        outs = self.module(inputs)
        loss_raw = F.cross_entropy(outs, targets, reduction="none")

        # reweighting
        if args.baseline:
            loss = torch.mean(loss_raw)
        else:
            logit = self.reweight(inputs)
            weight = torch.sigmoid(logit)
            loss = torch.mean(loss_raw * weight)  # / weight.detach().mean().item()

        return loss

    def configure_train_data_loader(self):
        return train_source_loader

    def configure_module(self):
        return build_model(num_classes=train_source_dataset.num_classes).to(device)

    def configure_optimizer(self):
        return build_optimizer(self.module, args)

    def configure_scheduler(self):
        return optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.step_size, gamma=args.gamma
        )


class Finetuning(ImplicitProblem):
    def forward(self, x):
        return self.module(x)

    def training_step(self, batch):
        inputs, targets, _ = batch
        outs = self(inputs)
        ce_loss = F.cross_entropy(outs, targets, reduction="none")
        ce_loss = torch.mean(ce_loss)
        reg_loss = self.reg_loss()

        return ce_loss + reg_loss

    def reg_loss(self):
        loss = 0
        for (n1, p1), (n2, p2) in zip(
            self.module.named_parameters(), self.pretrain.module.named_parameters()
        ):
            lam = 0 if "fc" in n1 else args.lam
            loss = loss + lam * (p1 - p2).pow(2).sum()
        return loss

    def configure_train_data_loader(self):
        return train_loader

    def configure_module(self):
        return build_model(num_classes=test_target_dataset.num_classes).to(device)

    def configure_optimizer(self):
        return build_optimizer(self.module, args)

    def configure_scheduler(self):
        return optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.step_size, gamma=args.gamma
        )


class Reweighting(ImplicitProblem):
    def forward(self, x):
        out = self.module(x).squeeze()
        return out

    def training_step(self, batch):
        inputs, targets, _ = batch
        outs = self.finetune(inputs)
        loss = F.cross_entropy(outs, targets)
        reg_loss = self.reg_loss()

        return loss + reg_loss

    def reg_loss(self):
        loss = 0
        for (n1, p1), (n2, p2) in zip(
            self.finetune.module.named_parameters(),
            self.pretrain.module.named_parameters(),
        ):
            lam = 0 if "fc" in n1 else args.lam
            loss = loss + lam * (p1 - p2).pow(2).sum()
        return loss

    def configure_train_data_loader(self):
        return valid_loader

    def configure_module(self):
        return build_model(num_classes=1).to(device)

    def configure_optimizer(self):
        return build_optimizer(self.module, args)

    def configure_scheduler(self):
        return optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.step_size, gamma=args.gamma
        )


best_acc = -1


class LBIEngine(Engine):
    @torch.no_grad()
    def validation(self):
        global best_acc
        correct = 0
        loss = 0
        total = 0
        for batch in test_loader:
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            outputs = self.finetune(inputs)
            loss += F.cross_entropy(outputs, targets, reduction="sum")
            correct += (outputs.argmax(dim=1) == targets).float().sum().item()
            total += inputs.size(0)
        acc = correct / total
        avgloss = loss / total
        if best_acc < acc:
            best_acc = acc

        return {"loss": avgloss, "acc": acc, "best_acc": best_acc}


# Define configs
reweight_config = Config(type="darts", retain_graph=True)
finetune_config = Config(type="darts", unroll_steps=1, allow_unused=False)
pretrain_config = Config(type="darts", unroll_steps=1, allow_unused=False)
engine_config = EngineConfig(valid_step=20, train_iters=1000, roll_back=False)

reweight = Reweighting(name="reweight", config=reweight_config, device=device)
finetune = Finetuning(name="finetune", config=finetune_config, device=device)
pretrain = Pretraining(name="pretrain", config=pretrain_config, device=device)
if args.baseline:
    problems = [finetune, pretrain]
else:
    problems = [reweight, finetune, pretrain]

if args.baseline:
    l2u = {pretrain: [finetune]}
    u2l = {}
else:
    u2l = {reweight: [pretrain]}
    l2u = {pretrain: [finetune, reweight], finetune: [reweight]}
dependencies = {"u2l": u2l, "l2u": l2u}

engine = LBIEngine(config=engine_config, problems=problems, dependencies=dependencies)
engine.run()
print("=" * 30)
print(f"{args.source_domain} --> {args.target_domain} || best_acc: {best_acc}")
print("=" * 30)
