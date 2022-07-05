import sys

sys.path.insert(0, "./..")

import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig


BASELINE = True
device = "cuda" if torch.cuda.is_available() else "cpu"


def build_dataset(reweight_size=1000, imbalanced_factor=100):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = MNIST(root="./data", train=True, download=True, transform=transform)

    num_classes = len(dataset.classes)
    num_meta = int(reweight_size / num_classes)

    index_to_meta = []
    index_to_train = []

    imbalanced_num_list = []
    sample_num = int((len(dataset.targets) - reweight_size) / num_classes)
    for class_index in range(num_classes):
        imbalanced_num = sample_num / (
            imbalanced_factor ** (class_index / (num_classes - 1))
        )
        imbalanced_num_list.append(int(imbalanced_num))
    np.random.shuffle(imbalanced_num_list)

    for class_index in range(num_classes):
        index_to_class = [
            index for index, label in enumerate(dataset.targets) if label == class_index
        ]
        np.random.shuffle(index_to_class)
        index_to_meta.extend(index_to_class[:num_meta])
        index_to_class_for_train = index_to_class[num_meta:]

        index_to_class_for_train = index_to_class_for_train[
            : imbalanced_num_list[class_index]
        ]

        index_to_train.extend(index_to_class_for_train)

    reweight_dataset = copy.deepcopy(dataset)
    dataset.data = dataset.data[index_to_train]
    dataset.targets = list(np.array(dataset.targets)[index_to_train])
    reweight_dataset.data = reweight_dataset.data[index_to_meta]
    reweight_dataset.targets = list(np.array(reweight_dataset.targets)[index_to_meta])

    return dataset, reweight_dataset


classifier_dataset, reweight_dataset = build_dataset(
    reweight_size=1000, imbalanced_factor=100
)


####################
##### Reweight #####
####################
reweight_dataloader = DataLoader(
    reweight_dataset, batch_size=100, shuffle=True, pin_memory=True
)
reweight_module = nn.Sequential(
    nn.Linear(1, 100), nn.ReLU(), nn.Linear(100, 1), nn.Sigmoid()
)
reweight_optimizer = optim.Adam(reweight_module.parameters(), lr=1e-5)


class Reweight(ImplicitProblem):
    def training_step(self, batch):
        inputs, labels = batch
        outputs = self.classifier(inputs)
        loss = F.cross_entropy(outputs, labels.long())

        return loss


reweight_config = Config()
reweight = Reweight(
    name="reweight",
    module=reweight_module,
    optimizer=reweight_optimizer,
    train_data_loader=reweight_dataloader,
    config=reweight_config,
    device=device,
)

####################
#### Classifier ####
####################
classifier_dataloader = DataLoader(
    classifier_dataset, batch_size=100, shuffle=True, pin_memory=True
)
classifier_module = nn.Sequential(
    nn.Flatten(), nn.Linear(784, 200), nn.ReLU(), nn.Linear(200, 10)
)
classifier_optimizer = optim.SGD(classifier_module.parameters(), lr=0.1, momentum=0.9)
classifier_scheduler = optim.lr_scheduler.MultiStepLR(
    classifier_optimizer, milestones=[1500, 2500], gamma=0.1
)


class Classifier(ImplicitProblem):
    def training_step(self, batch):
        inputs, labels = batch
        outputs = self.module(inputs)
        loss = F.cross_entropy(outputs, labels.long(), reduction="none")
        if BASELINE:
            return torch.mean(loss)
        loss_reshape = torch.reshape(loss, (-1, 1))
        weight = self.reweight(loss_reshape.detach())

        return torch.mean(weight * loss_reshape)


classifier_config = Config(type="darts", unroll_steps=1)
classifier = Classifier(
    name="classifier",
    module=classifier_module,
    optimizer=classifier_optimizer,
    scheduler=classifier_scheduler,
    train_data_loader=classifier_dataloader,
    config=classifier_config,
    device=device,
)


engine_config = EngineConfig(train_iters=3000)

if BASELINE:
    problems = [classifier]
    u2l, l2u = {}, {}
else:
    problems = [reweight, classifier]
    u2l = {reweight: [classifier]}
    l2u = {classifier: [reweight]}
dependencies = {"l2u": l2u, "u2l": u2l}

engine = Engine(config=engine_config, problems=problems, dependencies=dependencies)
engine.run()

# Validation
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
valid_dataset = MNIST(root="./data", train=False, transform=transform)
valid_dataloader = DataLoader(valid_dataset, batch_size=100, pin_memory=True)

correct = 0
total = 0
for x, target in valid_dataloader:
    x, target = x.to(device), target.to(device)
    out = classifier(x)
    correct += (out.argmax(dim=1) == target).sum().item()
    total += x.size(0)
acc = correct / total * 100
print("Classification Accuracy:", acc)
