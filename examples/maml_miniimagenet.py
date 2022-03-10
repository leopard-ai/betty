import argparse
import sys
sys.path.insert(0, "./..")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from betty.module import Module, HypergradientConfig
from betty.engine import Engine
from support.mini_imagenet import MiniImagenet


argparser = argparse.ArgumentParser()
argparser.add_argument('--n_way', type=int, help='n way', default=5)
argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
argparser.add_argument('--inner_steps', type=int, help='number of inner steps', default=5)
argparser.add_argument('--device', type=str, help='device', default='cuda')
argparser.add_argument('--task_num',type=int, help='meta batch size, namely task num', default=4)
argparser.add_argument('--seed', type=int, help='random seed', default=1)
arg = argparser.parse_args()

torch.manual_seed(arg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(arg.seed)
np.random.seed(arg.seed)

mini = MiniImagenet(
        '/home/ubuntu/workspace/datasets/mini-imagenet',
        batchsz=100,
        n_way=arg.n_way,
        k_shot=arg.k_spt,
        k_query=arg.k_qry,
        resize=84,
        mode='train'
    )
db = torch.utils.data.DataLoader(
    mini,
    arg.task_num,
    shuffle=True,
    pin_memory=True,
    num_workers=1
)

mini_test = MiniImagenet(
        '/home/ubuntu/workspace/datasets/mini-imagenet',
        batchsz=arg.task_num,
        n_way=arg.n_way,
        k_shot=arg.k_spt,
        k_query=arg.k_qry,
        resize=84,
        mode='test'
    )

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Net(nn.Module):
    def __init__(self, n_way, device, hidden_dim=32):
        super(Net, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(3, hidden_dim, 3),
                                 nn.BatchNorm2d(hidden_dim, momentum=1, affine=True),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2, 2),
                                 nn.Conv2d(hidden_dim, hidden_dim, 3),
                                 nn.BatchNorm2d(hidden_dim, momentum=1, affine=True),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2, 2),
                                 nn.Conv2d(hidden_dim, hidden_dim, 3),
                                 nn.BatchNorm2d(hidden_dim, momentum=1, affine=True),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2, 2),
                                 nn.Conv2d(hidden_dim, hidden_dim, 3),
                                 nn.BatchNorm2d(hidden_dim, momentum=1, affine=True),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2, 1),
                                 Flatten(),
                                 nn.Linear(hidden_dim*5*5, n_way)).to(device)

    def forward(self, x):
        return self.net.forward(x)


class Parent(Module):
    def forward(self, *args, **kwargs):
        return self.params, self.buffers

    def training_step(self, batch, *args, **kwargs):
        x_spt, y_spt, x_qry, y_qry = batch
        x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()
        loss = 0
        accs = []
        for idx, ch in enumerate(self._children):
            out = ch(self.batch[0][idx])
            loss += F.cross_entropy(out, self.batch[1][idx])
            accs.append((out.argmax(dim=1) == self.batch[1][idx]).detach())
        self.batch = (x_qry, y_qry)
        self.child_batch = (x_spt, y_spt)
        self.scheduler.step()
        if self.count % 10 == 0:
            acc = 100. * torch.cat(accs).float().mean().item()
            print('='*65)
            print('step:', self.count, '|| loss:', loss.clone().detach().item(), ' || acc:', acc)

        return loss

    def configure_train_data_loader(self):
        data_loader = iter(db)
        x_spt, y_spt, x_qry, y_qry = next(data_loader)
        x_spt, y_spt, x_qry, y_qry = x_spt.cuda(), y_spt.cuda(), x_qry.cuda(), y_qry.cuda()
        self.batch = (x_qry, y_qry)
        self.child_batch = (x_spt, y_spt)
        return data_loader

    def configure_module(self):
        return Net(arg.n_way, self.device)

    def configure_optimizer(self):
        return optim.Adam(self.module.parameters(), lr=0.001, betas=(0.5, 0.9))

    def configure_scheduler(self):
        return optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.95)


class Child(Module):
    def forward(self, x):
        return self.fmodule(self.params, self.buffers, x)

    def training_step(self, batch, *args, **kwargs):
        child_idx = self.parents[0].children.index(self)
        inputs, targets = self.parents[0].child_batch
        inputs, targets = inputs[child_idx], targets[child_idx]
        out = self.fmodule(self.params, self.buffers, inputs)
        loss = F.cross_entropy(out, targets)

        return loss

    def on_inner_loop_start(self):
        assert len(self._parents) == 1
        params, buffers = self._parents[0]()
        self.params = tuple(p.clone() for p in params)
        self.buffers = tuple(b.clone() for b in buffers)

    def configure_train_data_loader(self):
        return [None]

    def configure_module(self):
        return Net(arg.n_way, self.device)

    def configure_optimizer(self):
        return optim.SGD(self.module.parameters(), lr=0.01)


class MAMLEngine(Engine):
    def validation(self, data_loader):
        iters = data_loader.x_test.shape[0] // data_loader.batchsz
        data_loader = iter(data_loader)

        for _ in range(iters):
            x_spt, y_spt, x_qry, y_qry = next(data_loader)


parent_config = HypergradientConfig(type='maml',
                                    step=arg.inner_steps,
                                    first_order=False)
child_config = HypergradientConfig(type='maml',
                                   step=1,
                                   first_order=False,
                                   retain_graph=True)

parent = Parent(config=parent_config, device=arg.device)
children = [Child(config=child_config, device=arg.device) for _ in range(arg.task_num)]
problems = children + [parent]
dependencies = {parent: children}
engine = Engine(config=None, problems=problems, dependencies=dependencies)
engine.run()