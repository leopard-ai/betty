import sys

sys.path.insert(0, "./..")

import argparse
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig

parser = argparse.ArgumentParser(description="Distributed Training Tutorial")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--strategy", type=str, default="distributed")
args = parser.parse_args()


def build_dataset(reweight_size=1000, imbalanced_factor=100):
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )
    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)

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

    rwt_dataset = copy.deepcopy(dataset)
    dataset.data = dataset.data[index_to_train]
    dataset.targets = list(np.array(dataset.targets)[index_to_train])
    rwt_dataset.data = rwt_dataset.data[index_to_meta]
    rwt_dataset.targets = list(np.array(rwt_dataset.targets)[index_to_meta])

    return dataset, rwt_dataset


classifier_dataset, reweight_dataset = build_dataset(
    reweight_size=1000, imbalanced_factor=100
)
normalize = transforms.Normalize(
    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
)
valid_transform = transforms.Compose([transforms.ToTensor(), normalize])
valid_dataset = CIFAR10(root="./data", train=False, transform=valid_transform)
valid_dataloader = DataLoader(valid_dataset, batch_size=50, pin_memory=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


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
        acc = (outputs.argmax(dim=1) == labels.long()).float().mean().item() * 100

        return {"loss": loss, "acc": acc}


reweight_config = Config(log_step=100, fp16=True)
reweight = Reweight(
    name="reweight",
    module=reweight_module,
    optimizer=reweight_optimizer,
    train_data_loader=reweight_dataloader,
    config=reweight_config,
)

####################
#### Classifier ####
####################
classifier_dataloader = DataLoader(
    classifier_dataset, batch_size=50, shuffle=True, pin_memory=True
)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


classifier_module = ResNet50()
classifier_optimizer = optim.SGD(classifier_module.parameters(), lr=0.01, momentum=0.9)
classifier_scheduler = optim.lr_scheduler.MultiStepLR(
    classifier_optimizer, milestones=[6000, 8500], gamma=0.1
)


class Classifier(ImplicitProblem):
    def training_step(self, batch):
        inputs, labels = batch
        outputs = self.module(inputs)
        loss = F.cross_entropy(outputs, labels.long(), reduction="none")
        loss_reshape = torch.reshape(loss, (-1, 1))
        weight = self.reweight(loss_reshape.detach())

        return torch.mean(weight * loss_reshape)


classifier_config = Config(type="darts", unroll_steps=1, fp16=True)
classifier = Classifier(
    name="classifier",
    module=classifier_module,
    optimizer=classifier_optimizer,
    scheduler=classifier_scheduler,
    train_data_loader=classifier_dataloader,
    config=classifier_config,
)


class ReweightingEngine(Engine):
    @torch.no_grad()
    def validation(self):
        correct = 0
        total = 0
        if not hasattr(self, "best_acc"):
            self.best_acc = -1
        for x, target in valid_dataloader:
            x, target = x.to(device), target.to(device)
            out = self.classifier(x)
            correct += (out.argmax(dim=1) == target).sum().item()
            total += x.size(0)
        acc = correct / total * 100
        if self.best_acc < acc:
            self.best_acc = acc

        return {"acc": acc, "best_acc": self.best_acc}


engine_config = EngineConfig(
    train_iters=10000,
    valid_step=100,
    strategy=args.strategy,  # strategy in ["default", "distributed", "zero"]
)

problems = [reweight, classifier]
u2l = {reweight: [classifier]}
l2u = {classifier: [reweight]}
dependencies = {"l2u": l2u, "u2l": u2l}

engine = ReweightingEngine(
    config=engine_config, problems=problems, dependencies=dependencies
)
engine.run()
