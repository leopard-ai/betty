from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.optim as optim


def build_model(num_classes):
    model = resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    nn.init.xavier_uniform_(model.fc.weight, .1)
    nn.init.constant_(model.fc.bias, 0.)
    return model


def build_optimizer(model, args):
    optimizer = optim.Adam(
        [{'params': [param for name, param in model.named_parameters() if 'fc' not in name], 'lr': args.features_lr},
         {'params': model.fc.parameters(), 'lr': args.classifier_lr}], weight_decay=args.weight_decay
    )
    return optimizer