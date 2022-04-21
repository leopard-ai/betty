import random
import argparse
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils.utils import set_random_seed, disable_grads, enable_grads, evaluate
from utils.data_transform import transform
from dalib.vision.datasets import Office31, OfficeHome
from model.resnet import Resnet, param_lr, getParam, getOptim


domainIdxDict = {'Ar': 0, 'Cl': 1, 'Pr': 2, 'Rw': 3, 'A': 0, 'D': 1, 'W': 2}


def argument_parser():
    parser = argparse.ArgumentParser(description='regularize the target by the source')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--gpu', type=int, help='GPU idx to run', default=0)
    parser.add_argument('--model', type=str, default='resnet')
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

train_source_loader = torch.utils.data.DataLoader(train_source_dataset,
                                                  batch_size=args.batch_size,
                                                  num_workers=6,
                                                  shuffle=True,
                                                  pin_memory=True,
                                                  drop_last=True)

if args.model == 'resnet':
    print('Using resnet')
    model_tgt = Resnet(num_classes=train_target_dataset.num_classes).to(device)
    optimizer_tgt = getOptim(model_tgt, args)
    scheduler_tgt = optim.lr_scheduler.StepLR(optimizer_tgt, step_size=args.step_size, gamma=1e-1)

    model_src = Resnet(num_classes=train_source_dataset.num_classes).to(device)
    optimizer_src = getOptim(model_src, args)
    scheduler_src = optim.lr_scheduler.StepLR(optimizer_src, step_size=args.step_size, gamma=1e-1)