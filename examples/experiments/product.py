import sys
sys.path.insert(0, "./../..")

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from betty.engine import Engine
from betty.config_template import Config
from betty.problems import ImplicitProblem, IterativeProblem


batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
dataset2 = datasets.MNIST('../data', train=False, transform=transform)
num_train = len(dataset1) # 50000
indices = list(range(num_train))
split = 5000
train_kwargs = {'batch_size': batch_size,
                'sampler': torch.utils.data.sampler.SubsetRandomSampler(indices[:split])}
test_kwargs = {'batch_size': 100}
if device == 'cuda':
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
train_kwargs2 = copy.deepcopy(train_kwargs)
train_kwargs2['sampler'] = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


class Parent(nn.Module):
    def __init__(self):
        super(Parent, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, x2):
        x = torch.concat((x, x2), dim=1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class Outer(ImplicitProblem):
    def forward(self, x):
        return self.module( x)

    def training_step(self, batch, *args, **kwargs):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = self(inputs)
        outputs = self.inner(inputs, outputs)
        loss = F.nll_loss(outputs, targets)
        self.scheduler.step()

        return loss

    def configure_train_data_loader(self):
        return torch.utils.data.DataLoader(dataset1, **train_kwargs)

    def configure_module(self):
        return Parent().to(device)

    def configure_optimizer(self):
        #return optim.SGD(self.module.parameters(), lr=0.01, momentum=0.9)
        return optim.Adam(self.module.parameters(), lr=0.0005)

    def configure_scheduler(self):
        return StepLR(self.optimizer, step_size=125, gamma=0.9, verbose=False)


class Inner(ImplicitProblem):
    def forward(self, inputs, inputs2):
        return self.module(inputs, inputs2)

    def training_step(self, batch, *args, **kwargs):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = self.outer(inputs)
        outputs = self(inputs, outputs)
        loss = F.nll_loss(outputs, targets)
        self.scheduler.step()
        return loss

    def configure_train_data_loader(self):
        return torch.utils.data.DataLoader(dataset1, **train_kwargs2)

    def configure_module(self):
        return Net().to(device)

    def configure_optimizer(self):
        #return optim.SGD(self.module.parameters(), lr=0.01, momentum=0.9)
        return optim.Adam(self.module.parameters(), lr=0.001)

    def configure_scheduler(self):
        return StepLR(self.optimizer, step_size=3000, gamma=0.9, verbose=False)


outer_config = Config(type='neumann',
                      step=20,
                      neumann_alpha=0.001,
                      neumann_iterations=10,
                      retain_graph=True,
                      first_order=True,
                      allow_unused=False)
inner_config = Config(type='torch',
                      allow_unused=False)
outer = Outer(name='outer', config=outer_config, device=device)
inner = Inner(name='inner', config=inner_config, device=device)

problems = [outer, inner]
dependencies = {outer: [inner]}
#dependencies2 = {inner: [outer]}


class MyEngine(Engine):
    def validation(self):
        print('Validation')
        correct = 0
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = self.outer(inputs)
            outputs = self.inner(inputs, outputs)
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

        print('Test Acc.:', 100. * correct / len(test_loader.dataset))

    def train_step(self):
    #    self.set_dependency(dependencies)
        for leaf in self.leaves:
            leaf.step(param_update=False)
    #    self.set_dependency(dependencies2)
    #    for leaf in self.leaves:
    #        leaf.step(param_update=False)

engine = MyEngine(config=None, problems=problems, dependencies=dependencies)
engine.run()
