from os import device_encoding
import sys
from xml.dom.pulldom import default_bufsize
sys.path.insert(0, "./../..")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem


BASELINE =  False
default_wdecay = 5e-4 if BASELINE else 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
trainset = datasets.MNIST('../data', train=True, download=True, transform=transform)
testset = datasets.MNIST('../data', train=False, transform=transform)
num_train = len(trainset) # 50000
indices = list(range(num_train))
split = int(1. * num_train)
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
net_optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=default_wdecay)
train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    num_workers=1,
    shuffle=True,
    pin_memory=True,
)

class Classifier(ImplicitProblem):
    def training_step(self, batch):
        x, target = batch
        out = self.module(x)
        ce_loss = F.cross_entropy(out, target)
        if BASELINE:
            return ce_loss
        else:
            fc1_wdecay, fc2_wdecay = self.hpo()
            reg_loss = torch.sum(torch.pow(self.module.fc1.weight, 2) * fc1_wdecay) / 2 + \
                torch.sum(torch.pow(self.module.fc2.weight, 2) * fc2_wdecay) / 2

            return ce_loss + reg_loss

classifier_config = Config(type='darts', step=1, first_order=True)
classifier = Classifier(
    name='classifier',
    module=net,
    optimizer=net_optimizer,
    train_data_loader=train_loader,
    config=classifier_config,
    device=device
)

# Outer problem
class WeightDecay(nn.Module):
    def __init__(self):
        super(WeightDecay, self).__init__()
        self.fc1_wdecay = nn.Parameter(torch.ones(200, 784) * 5e-4)
        self.fc2_wdecay = nn.Parameter(torch.ones(10, 200) * 5e-4)

    def forward(self):
        return self.fc1_wdecay, self.fc2_wdecay

wdecay = WeightDecay()
wdecay_optimizer = optim.Adam(wdecay.parameters(), lr=1e-5)
valid_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
)

class HPO(ImplicitProblem):
    def training_step(self, batch):
        x, target = batch
        out = self.classifier(x)
        loss = F.cross_entropy(out, target)
        fc1_wdecay, fc2_wdecay = self()
        reg_loss = torch.sum(torch.pow(self.classifier.module.fc1.weight, 2) * fc1_wdecay) / 2 + \
                torch.sum(torch.pow(self.classifier.module.fc2.weight, 2) * fc2_wdecay) / 2
        acc = (out.argmax(dim=1) == target.long()).float().mean().item() * 100
        loss = loss + reg_loss

        return {'loss': loss, 'acc': acc}

    def param_callback(self, params):
        for p in params:
            p.data.clamp_(min=1e-8)

hpo_config = Config(type='darts', step=1, first_order=True, retain_graph=True)
hpo = HPO(
    name='hpo',
    module=wdecay,
    optimizer=wdecay_optimizer,
    train_data_loader=valid_loader,
    config=hpo_config,
    device=device
)

best_acc = -1
class HPOEngine(Engine):
    @torch.no_grad()
    def validation(self):
        correct = 0
        total = 0
        global best_acc
        for x, target in test_loader:
            x, target = x.to(device), target.to(device)
            with torch.no_grad():
                out = self.classifier(x)
            correct += (out.argmax(dim=1) == target).sum().item()
            total += x.size(0)
        acc = correct / total * 100
        if best_acc < acc:
            best_acc = acc
        return {'acc': acc, 'best_acc': best_acc}


    def train_step(self):
        for leaf in self.leaves:
            leaf.step(param_update=False)

if BASELINE:
    problems = [classifier]
    u2l = {}
    l2u = {}
else:
    problems = [classifier, hpo]
    u2l = {hpo: [classifier]}
    l2u = {classifier: [hpo]}
dependencies = {'l2u': l2u, 'u2l': u2l}

engine_config = EngineConfig(train_iters=5000, valid_step=100)
engine = HPOEngine(config=engine_config, problems=problems, dependencies=dependencies)
engine.run()
