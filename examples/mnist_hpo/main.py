import sys
sys.path.insert(0, "./../..")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from betty.engine import Engine
from betty.configs import Config
from betty.problems import ImplicitProblem


transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
trainset = datasets.MNIST('../data', train=True, download=True, transform=transform)
testset = datasets.MNIST('../data', train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=1)

# Inner problem
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 200, bias=False)
        self.fc2 = nn.Linear(200, 10, bias=False)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        return out

net = Net()
net_optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1)

class Classifier(ImplicitProblem):
    def forward(self, x):
        return self.module(x)

    def training_step(self, batch):
        x, target = batch
        out = self.module(x)
        ce_loss = F.cross_entropy(out, target)
        fc1_wdecay, fc2_wdecay = self.hpo()
        reg_loss = torch.sum(torch.pow(self.module.fc1.weight, 2) * fc1_wdecay) / 2 + \
            torch.sum(torch.pow(self.module.fc2.weight, 2) * fc2_wdecay) / 2

        return ce_loss + reg_loss

classifier_config = Config(type='darts', step=1, first_order=True)
classifier = Classifier(
    name='classifier',
    module=net,
    optiimzer=net_optimizer,
    train_data_loader=train_loader,
    config=classifier_config
)

# Outer problem
class WeightDecay(nn.Module):
    def __init__(self):
        super(WeightDecay, self).__init__()
        self.fc1_wdecay = nn.Parameter(torch.ones(784, 200) * 5e-4)
        self.fc2_wdecay = nn.Parameter(torch.ones(200, 100) * 5e-4)

    def forward(self):
        return self.fc1_wdecay, self.fc2_wdecay

wdecay = WeightDecay()
wdecay_optimizer = optim.Adam(wdecay.parameters(), lr=1e-4)
valid_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=1)

class HPO(ImplicitProblem):
    def forward(self):
        return self.module()

    def training_step(self, batch):
        x, target = batch
        out = self.classifier(x)

        return F.cross_entropy(out, target)

    def param_callback(self, params):
        for p in params:
            p.data.clamp_(min=1e-8)

hpo_config = Config(type='darts', log_step=10, step=1, first_order=True)
hpo = HPO(
    name='hpo',
    module=wdecay,
    optimizer=wdecay_optimizer,
    train_data_loader=valid_loader,
    config=hpo_config
)


problems = [classifier, hpo]
u2l = {hpo: [classifier]}
l2u = {classifier: [hpo]}
dependencies = {'l2u': l2u, 'u2l': u2l}

engine = Engine(config=None, problems=problems, dependencies=dependencies)
engine.run()
