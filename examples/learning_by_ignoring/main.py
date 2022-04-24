import sys
sys.path.insert(0, "./../..")
import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim

from betty.engine import Engine
from betty.problems import ImplicitProblem, IterativeProblem
from betty.config_template import Config

from utils.utils import set_random_seed, evaluate
from utils.data_transform import transform
from dalib.vision.datasets import Office31, OfficeHome
from model.resnet import Resnet, param_lr, getParam, getOptim


domainIdxDict = {'Ar': 0, 'Cl': 1, 'Pr': 2, 'Rw': 3, 'A': 0, 'D': 1, 'W': 2}


def argument_parser():
    parser = argparse.ArgumentParser(description='regularize the target by the source')
    parser.add_argument('--gpu', type=int, help='GPU idx to run', default=0)
    parser.add_argument('--source_domain', type=str, default="Cl")
    parser.add_argument('--target_domain', type=str, default="Ar")
    parser.add_argument('--features_lr', type=float, default=1e-4)
    parser.add_argument('--classifier_lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='officehome')
    parser.add_argument('--data_dir', type=str, metavar='PATH', default='./data')
    parser.add_argument('--lam', type=float, help='lambda', default=7e-3)
    parser.add_argument('--gamma', type=float, help='gamma', default=1)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--factor', type=float, default=0.1)
    parser.add_argument('--meta_loop', type=int, default=1)
    parser.add_argument('--step_size', type=int, default=40)

    return parser

parser = argument_parser()
args = parser.parse_args()

set_random_seed(args.random_seed)
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# Dataset
train_transform = transform(train=True)
test_transform = transform(train=False)
data_root = os.path.join(args.data_dir, args.dataset)
if args.dataset == 'office31':
    getDataset = Office31
elif args.dataset == 'officehome':
    getDataset = OfficeHome

train_source_dataset = getDataset(root=data_root,
                                  task=args.source_domain + '_train',
                                  download=True,
                                  transform=train_transform)

print(f'train target_task: {args.target_domain}')
train_target_dataset = getDataset(root=data_root,
                                  task=args.target_domain + '_train',
                                  download=True,
                                  transform=train_transform)
valid_target_dataset = getDataset(root=data_root,
                                  task=args.target_domain + '_val',
                                  download=True,
                                  transform=test_transform)
test_target_dataset = getDataset(root=data_root,
                                 task=args.target_domain + '_test',
                                 download=True,
                                 transform=test_transform)


class DatasetwithIndice(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __getitem__(self, index):
        data, target, domain_idx = self.base[index]
        return data, target, domain_idx, index

    def __len__(self):
        return len(self.base)


class DataWeights(torch.nn.Module):
    def __init__(self, dim):
        super(DataWeights, self).__init__()

        self.weights = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, indexes):
        return self.weights[indexes]


train_source_dataset_patch = DatasetwithIndice(train_source_dataset)

train_loader = torch.utils.data.DataLoader(train_target_dataset,
                                           batch_size=args.batch_size,
                                           num_workers=6,
                                           shuffle=True,
                                           pin_memory=True,
                                           drop_last=True)

valid_loader = torch.utils.data.DataLoader(valid_target_dataset,
                                           batch_size=args.batch_size,
                                           num_workers=6,
                                           shuffle=True,
                                           pin_memory=True,
                                           drop_last=False)
test_loader = torch.utils.data.DataLoader(test_target_dataset,
                                          batch_size=args.batch_size,
                                          num_workers=6,
                                          shuffle=False,
                                          pin_memory=True,
                                          drop_last=False)

train_source_loader = torch.utils.data.DataLoader(train_source_dataset_patch,
                                                  batch_size=args.batch_size,
                                                  num_workers=6,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  drop_last=True)

model_tgt = Resnet(num_classes=train_target_dataset.num_classes).to(device)
optimizer_tgt = getOptim(model_tgt, args)
scheduler_tgt = optim.lr_scheduler.StepLR(optimizer_tgt, step_size=args.step_size, gamma=1e-1)

model_src = Resnet(num_classes=train_source_dataset.num_classes).to(device)
optimizer_src = getOptim(model_src, args)
scheduler_src = optim.lr_scheduler.StepLR(optimizer_src, step_size=args.step_size, gamma=1e-1)

dataweight = DataWeights(len(train_source_dataset)).to(device)


class Pretraining(ImplicitProblem):
    def forward(self, x):
        return self.module(x)

    def training_step(self, batch):
        inputs = batch[0].to(device)
        targets = batch[1].to(device)
        outs = self.module(inputs)
        loss_raw = F.cross_entropy(outs, targets, reduction='none')

        # reweighting
        indexes = batch[3]
        weights = torch.sigmoid(self.reweight(indexes))
        scale = weights.sum().detach().item()
        loss = torch.dot(loss_raw, weights) / scale

        return loss

    def configure_train_data_loader(self):
        return train_source_loader

    def configure_module(self):
        return model_src

    def configure_optimizer(self):
        return optimizer_src

    def configure_scheduler(self):
        return scheduler_src


class Finetuning(ImplicitProblem):
    def forward(self, x):
        return self.module(x)

    def training_step(self, batch):
        inputs = batch[0].to(device)
        targets = batch[1].to(device)
        outs = self.module(inputs)
        ce_loss = F.cross_entropy(outs, targets)

        reg_loss = 0
        for p1, p2 in zip(self.module.parameters(), self.pretrain.parameters()):
            reg_loss += (p1 - p2).pow(2).sum()

        return ce_loss + args.lam * reg_loss

    def configure_train_data_loader(self):
        return train_loader

    def configure_module(self):
        return model_tgt

    def configure_optimizer(self):
        return optimizer_tgt

    def configure_scheduler(self):
        return scheduler_tgt


class Reweighting(ImplicitProblem):
    def forward(self, indexes):
        return self.module(indexes)

    def training_step(self, batch):
        inputs = batch[0].to(device)
        targets = batch[1].to(device)
        outs = self.finetune(inputs)
        loss = F.cross_entropy(outs, targets)

        return loss

    def configure_train_data_loader(self):
        return valid_loader

    def configure_module(self):
        return dataweight

    def configure_optimizer(self):
        return optim.Adam(self.module.parameters(), lr=0.01, betas=(0.5, 0.999))
        

class LBIEngine(Engine):
    def validation(self):
        correct = 0
        total = 0
        for batch in test_loader:
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            outputs = self.finetune(inputs)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += inputs.size(0)
        acc = correct / total
        return acc


# Define configs
reweight_config = Config(type='darts',
                            step=1,
                            retain_graph=True,
                            first_order=True)
finetune_config = Config(type='torch', step=1)
pretrain_config = Config(type='torch', step=1)

reweight = Reweighting(name='reweight', config=reweight_config, device=device)
finetune = Finetuning(name='finetune', config=finetune_config, device=device)
pretrain = Pretraining(name='pretrain', config=pretrain_config, device=device)
problems = [reweight, finetune, pretrain]

h2l = {reweight: [pretrain]}
l2h = {pretrain: [finetune], finetune: [reweight]}
dependencies = {'h2l': h2l, 'l2h': l2h}

engine = LBIEngine(config=None, problems=problems, dependencies=dependencies)
engine.run()
