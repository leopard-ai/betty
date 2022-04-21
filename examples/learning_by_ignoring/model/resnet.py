from torchvision.models import resnet50, resnet34, resnet18
import torch
import torch.nn as nn
import torch.nn.init as init
import copy


def param_lr(model, args):
    lr_groups = []
    lr_groups.extend([(model.base_model.conv1.parameters(), args.features_lr),
                      (model.base_model.bn1.parameters(), args.features_lr),
                      (model.base_model.layer1.parameters(), args.features_lr),
                      (model.base_model.layer2.parameters(), args.features_lr),
                      (model.base_model.layer3.parameters(), args.features_lr),
                      (model.base_model.layer4.parameters(), args.features_lr),
                      (model.base_model.fc.parameters(), args.classifier_lr)])
    return lr_groups


def getParam(model):

    return list(model.base_model.conv1.parameters()) + list(
        model.base_model.bn1.parameters()) + list(
            model.base_model.layer1.parameters()) + list(
                model.base_model.layer2.parameters()) + list(
                    model.base_model.layer3.parameters()) + list(
                        model.base_model.layer4.parameters()) + list(
                            model.base_model.fc.parameters())


def getOptim(model, args):
    optim = torch.optim.Adam([{
        'params': model.base_model.conv1.parameters(),
        'lr': args.features_lr
    }, {
        'params': model.base_model.bn1.parameters(),
        'lr': args.features_lr
    }, {
        'params': model.base_model.layer1.parameters(),
        'lr': args.features_lr
    }, {
        'params': model.base_model.layer2.parameters(),
        'lr': args.features_lr
    }, {
        'params': model.base_model.layer3.parameters(),
        'lr': args.features_lr
    }, {
        'params': model.base_model.layer4.parameters(),
        'lr': args.features_lr
    }, {
        'params': model.base_model.fc.parameters(),
        'lr': args.classifier_lr
    }],
                             lr=args.features_lr,
                             weight_decay=args.weight_decay)
    return optim


def base_resnet(num_classes):
    model = resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    nn.init.xavier_uniform_(model.fc.weight, .1)
    nn.init.constant_(model.fc.bias, 0.)
    return model


class Resnet(nn.Module):
    def __init__(self, num_classes):

        super(Resnet, self).__init__()
        base_model = base_resnet(num_classes=num_classes)
        self.base_model = base_model

        self.conv_features = torch.nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu,
            base_model.maxpool, base_model.layer1, base_model.layer2,
            base_model.layer3, base_model.layer4, base_model.avgpool,
            torch.nn.Flatten(start_dim=1))
        # No dense features really
        self.dense_features = torch.nn.Identity()

        self.features = torch.nn.Sequential(self.conv_features,
                                            self.dense_features)

        self.output = torch.nn.Sequential(self.features, base_model.fc)
        self.parameter = list(self.features.parameters()) + list(
            self.output.parameters())

    def forward(self, x):
        # x = self.base_model.conv1(x)
        # x = self.base_model.bn1(x)
        # x = self.base_model.relu(x)
        # x = self.base_model.maxpool(x)

        # x = self.base_model.layer1(x)
        # x = self.base_model.layer2(x)
        # x = self.base_model.layer3(x)
        # x = self.base_model.layer4(x)

        # x = self.base_model.avgpool(x)
        # x = x.view(x.size(0), -1)
        # output_class = self.base_model.fc(x)
        # return output_class
        return self.output(x)

    def classifier(self, x):
        x = self.base_model.fc(x)
        return x
