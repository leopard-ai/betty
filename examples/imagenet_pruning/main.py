import os
import json
import argparse
from pathlib import Path

import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig

from dataset import ImageNet, get_subset_data

# from util import load_data
from model import resnet18, resnet50, MLP
from util import Summary, AverageMeter, ProgressMeter


def get_args() -> argparse.Namespace:
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch ResNet50 Example")
    parser.add_argument("--layers", type=int, default=50)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=1,
        help="gradient accumulation",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="SGD weight_decay (default: 0.001)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="training precision",
    )
    parser.add_argument(
        "--baseline", action="store_true", default=False, help="disables meta learning"
    )
    parser.add_argument(
        "--strategy", type=str, default="default", help="(Distributed) trainin strategy"
    )
    parser.add_argument(
        "--nesterov",
        action="store_true",
        default=False,
        help="enables nesterov momentum",
    )
    parser.add_argument(
        "--prune", action="store_true", default=False, help="dataset pruning"
    )
    parser.add_argument(
        "--prune_strategy",
        type=str,
        default="random",
        help="Pruning strategy. Choices: metaweight, random",
    )
    parser.add_argument(
        "--frac_data_kept",
        type=float,
        default=1.0,
        help="Fraction of data kept",
    )
    parser.add_argument(
        "--instance_weights_dir",
        type=str,
        default="",
        help="Directory where instance weights are stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--unroll_step", type=int, default=5, metavar="S", help="Log step"
    )
    parser.add_argument(
        "--log_step", type=int, default=100, metavar="S", help="Log step"
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        type=str,
        help="Location of the dataset (will be downloaded if needed)",
    )
    parser.add_argument(
        "--interpolation",
        default="bilinear",
        type=str,
        help="the interpolation method (default: bilinear)",
    )
    parser.add_argument(
        "--val-resize-size",
        default=256,
        type=int,
        help="the resize size used for validation (default: 256)",
    )
    parser.add_argument(
        "--val-crop-size",
        default=224,
        type=int,
        help="the central crop size used for validation (default: 224)",
    )
    parser.add_argument(
        "--train-crop-size",
        default=224,
        type=int,
        help="the random crop size used for training (default: 224)",
    )
    parser.add_argument("--imagenet-classes", default="metadata/imagenet_classes.json")
    parser.add_argument("--checkpoint_directory", type=str, default=".")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Do not specify this argument. torch.distributed.launch sets this value. "
        "Rank 0 handles all main thread operations like logging and checkpointing.",
    )

    return parser.parse_args()


args = get_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Data loading
with open(args.imagenet_classes) as f:
    classes = list(json.load(f)["classes"])


if args.prune:
    dataset = ImageNet(
        dataset_file=args.data_dir,
        sample_set="train",
        classes=classes,
        transform_type="train",
        args=args,
    )
    print("Dataset pruning!")

    train_dataset = get_subset_data(
        dataset=dataset,
        prune_strategy=args.prune_strategy,
        instance_weights_dir=args.instance_weights_dir,
        frac_data_kept=args.frac_data_kept,
    )
    print("No. of examples (after pruning): ", len(train_dataset))
else:
    train_dataset = ImageNet(
        dataset_file=args.data_dir,
        sample_set="train",
        classes=classes,
        transform_type="train",
        args=args,
    )

test_dataset = ImageNet(
    dataset_file=args.data_dir,
    sample_set="val",
    classes=classes,
    transform_type="val",
    args=args,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
)
meta_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)

if args.layers == 18:
    model = resnet18()
else:
    model = resnet50()

optimizer = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    nesterov=True if args.nesterov else False,
)

meta_model = MLP(2, 100, 1)
meta_optimizer = optim.Adam(meta_model.parameters(), lr=1e-5)


# iterations
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

print("Per device batch size : ", args.batch_size)
print("Gradient accumulation : ", args.gradient_accumulation)
print("World size : ", WORLD_SIZE)
print(
    "Effective batch size : ", args.batch_size * args.gradient_accumulation * WORLD_SIZE
)

epoch_iter = int(
    len(train_dataset) / (WORLD_SIZE * args.batch_size * args.gradient_accumulation)
)
total_iter = int(epoch_iter * args.epochs)
decay_iter1 = int(epoch_iter * 40)
decay_iter2 = int(epoch_iter * 80)
print("epoch iter:", epoch_iter)
print("total iter:", total_iter)
print("decay iter1:", decay_iter1)
print("decay iter2:", decay_iter2)
print("weight decay:", args.weight_decay)

scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[decay_iter1, decay_iter2],
    gamma=0.1,
)

checkpoint_directory = args.checkpoint_directory
os.makedirs(checkpoint_directory, exist_ok=True)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Outer(ImplicitProblem):
    def training_step(self, batch):
        inputs, labels = batch
        outputs, *_ = self.inner(inputs)
        loss = F.cross_entropy(outputs, labels.long())

        return loss


class Inner(ImplicitProblem):
    def training_step(self, batch):
        inputs, labels = batch
        outputs, ema_outputs = self.forward(inputs)

        # loss calculation
        if args.baseline:
            return F.cross_entropy(outputs, labels.long())
        loss_vector = F.cross_entropy(outputs, labels.long(), reduction="none")
        loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))

        # ema part
        ema_prob = F.softmax(ema_outputs, dim=-1)
        ema_loss_vector = torch.sum(-F.log_softmax(outputs, dim=-1) * ema_prob, dim=-1)
        ema_loss_vector_reshape = torch.reshape(ema_loss_vector, (-1, 1))

        # reweighting
        meta_inputs = torch.cat(
            [loss_vector_reshape.detach(), ema_loss_vector_reshape.detach()], dim=1
        )
        weight = self.outer(meta_inputs, self._global_step)
        loss = torch.mean(weight * loss_vector_reshape)

        return loss

    def param_callback(self):
        if args.strategy == "default":
            self.module.ema_update()
        else:
            self.module.module.ema_update()


class ReweightingEngine(Engine):
    @torch.no_grad()
    def validation(self):
        correct = 0
        total = 0
        if not hasattr(self, "best_acc"):
            self.best_acc = -1
        if not hasattr(self, "best_acc1"):
            self.best_acc1 = -1
        if not hasattr(self, "best_acc5"):
            self.best_acc5 = -1

        top1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
        top5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)

        # validation loop
        for x, target in test_loader:
            x, target = x.to(self.inner.device), target.to(self.inner.device)
            with torch.no_grad():
                out, *_ = self.inner(x)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(out, target, topk=(1, 5))

            top1.update(acc1[0], x.size(0))
            top5.update(acc5[0], x.size(0))

            correct += (out.argmax(dim=1) == target).sum().item()
            total += x.size(0)

        acc = correct / total * 100

        # best acc update
        if self.best_acc < acc:
            self.best_acc = acc

        # Here we use top-5 metric for best model selection
        if self.best_acc5 < top5.avg:
            self.best_acc5 = top5.avg
            self.best_acc1 = top1.avg

        elif self.best_acc5 == top5.avg:
            if self.best_acc1 <= top1.avg:
                self.best_acc1 = top1.avg

        # save
        classifier = self.inner.module
        if not args.baseline:
            reweighter = self.outer.module
        if args.strategy != "default":
            classifier = classifier.module
            if not args.baseline:
                reweighter = reweighter.module
        torch.save(
            classifier.state_dict(),
            "{}/cls_{}.pt".format(checkpoint_directory, self.global_step),
        )
        if not args.baseline:
            torch.save(
                reweighter.state_dict(),
                "{}/mwn_{}.pt".format(checkpoint_directory, self.global_step),
            )

        return {
            "acc": acc,
            "best_acc": self.best_acc,
            "Acc@1": top1.avg,
            "Best_Acc@1": self.best_acc1,
            "Acc@5": top5.avg,
            "Best_Acc@5": self.best_acc5,
        }


# config
inner_log_step = -1 if not args.baseline else args.log_step
outer_config = Config(
    precision=args.precision,
    log_step=int(args.log_step // args.unroll_step),
    retain_graph=True,
)
inner_config = Config(
    type="darts",
    precision=args.precision,
    log_step=inner_log_step,
    unroll_steps=args.unroll_step,
    gradient_accumulation=args.gradient_accumulation,
)
engine_config = EngineConfig(
    train_iters=total_iter,
    valid_step=epoch_iter,
    strategy=args.strategy,
)

# problems
outer = Outer(
    name="outer",
    module=meta_model,
    optimizer=meta_optimizer,
    train_data_loader=meta_loader,
    config=outer_config,
)
inner = Inner(
    name="inner",
    module=model,
    optimizer=optimizer,
    scheduler=scheduler,
    train_data_loader=train_loader,
    config=inner_config,
)


# engine
problems = [inner]
u2l, l2u = {}, {}
if not args.baseline:
    problems = [inner, outer]
    u2l = {outer: [inner]}
    l2u = {inner: [outer]}
dependencies = {"l2u": l2u, "u2l": u2l}

engine = ReweightingEngine(
    config=engine_config, problems=problems, dependencies=dependencies
)
engine.run()
