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
from support.omniglot_loader import OmniglotNShot


argparser = argparse.ArgumentParser()
argparser.add_argument('--n_way', type=int, help='n way', default=5)
argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
argparser.add_argument('--inner_steps', type=int, help='number of inner steps', default=50)
argparser.add_argument('--device', type=str, help='device', default='cuda')
argparser.add_argument('--task_num',type=int, help='meta batch size, namely task num', default=16)
argparser.add_argument('--seed', type=int, help='random seed', default=1)
arg = argparser.parse_args()

torch.manual_seed(arg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(arg.seed)
np.random.seed(arg.seed)

db = OmniglotNShot(
        '/tmp/omniglot-data',
        batchsz=arg.task_num,
        n_way=arg.n_way,
        k_shot=arg.k_spt,
        k_query=arg.k_qry,
        imgsz=28,
        device=arg.device,
    )

db_test = OmniglotNShot(
        '/tmp/omniglot-data-test',
        batchsz=arg.task_num,
        n_way=arg.n_way,
        k_shot=arg.k_spt,
        k_query=arg.k_qry,
        imgsz=28,
        device=arg.device,
        mode='test'
    )

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Net(nn.Module):
    def __init__(self, n_way, device):
        super(Net, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 64, 3),
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
                                 Flatten(),
                                 nn.Linear(64, n_way)).to(device)

    def forward(self, x):
        return self.net.forward(x)


class Parent(Module):
    def forward(self, *args, **kwargs):
        return self.params, self.buffers

    def training_step(self, batch, *args, **kwargs):
        x_spt, y_spt, x_qry, y_qry = batch
        losses = []
        accs = []
        for idx in range(len(self._children)):
            ch = getattr(self, f'inner_{idx}')
            out = ch(self.batch[0][idx])
            loss = F.cross_entropy(out, self.batch[1][idx])
            losses.append(loss)
            accs.append((out.argmax(dim=1) == self.batch[1][idx]).detach())
        self.batch = (x_qry, y_qry)
        self.child_batch = (x_spt, y_spt)
        #self.scheduler.step()
        if self.count % 10 == 0:
            acc = 100. * torch.cat(accs).float().mean().item()
            print('step:', self.count, '|| loss:', sum(losses).clone().detach().item(), ' || acc:', acc)

        return losses

    def configure_train_data_loader(self):
        data_loader = db
        x_spt, y_spt, x_qry, y_qry = next(data_loader)
        self.batch = (x_qry, y_qry)
        self.child_batch = (x_spt, y_spt)
        return data_loader

    def configure_module(self):
        return Net(arg.n_way, self.device)

    def configure_optimizer(self):
        return optim.Adam(self.module.parameters(), lr=0.001)
        #return optim.SGD(self.module.parameters(), lr=0.01)

    #def configure_scheduler(self):
    #    return optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.9)


class Child(Module):
    def forward(self, x):
        return self.fmodule(self.params, self.buffers, x)

    def training_step(self, batch, *args, **kwargs):
        child_idx = self.outer.children.index(self)
        inputs, targets = self.outer.child_batch
        inputs, targets = inputs[child_idx], targets[child_idx]
        out = self.fmodule(self.params, self.buffers, inputs)
        loss = F.cross_entropy(out, targets)

        return loss

    def on_inner_loop_start(self):
        assert len(self._parents) == 1
        params, buffers = self.outer()
        self.params = tuple(p.clone() for p in params)
        self.buffers = tuple(b.clone() for b in buffers)

    def configure_train_data_loader(self):
        return [None]

    def configure_module(self):
        return Net(arg.n_way, self.device)

    def configure_optimizer(self):
        return optim.SGD(self.module.parameters(), lr=0.1)


class MAMLEngine(Engine):
    def validation(self, data_loader):
        iters = data_loader.x_test.shape[0] // data_loader.batchsz
        data_loader = iter(data_loader)

        for _ in range(iters):
            x_spt, y_spt, x_qry, y_qry = next(data_loader)


parent_config = HypergradientConfig(type='darts',
                                    step=arg.inner_steps,
                                    retain_graph=True,
                                    first_order=True)
child_config = HypergradientConfig(type='maml',
                                   step=1,
                                   first_order=False,
                                   retain_graph=True)

parent = Parent(name='outer', config=parent_config, device=arg.device)
children = [Child(name='inner', config=child_config, device=arg.device) for _ in range(arg.task_num)]
problems = children + [parent]
dependencies = {parent: children}
engine = Engine(config=None, problems=problems, dependencies=dependencies)
engine.run()
