from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.optim as optim


def build_model(num_classes):
    model = resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    nn.init.xavier_uniform_(model.fc.weight, 0.1)
    nn.init.constant_(model.fc.bias, 0.0)
    return model


def build_optimizer(model, args, betas=None, weight_decay=None, lrs=None):
    wd = weight_decay if weight_decay is not None else args.weight_decay
    b = betas if betas is not None else (0.9, 0.999)
    lr1 = lrs[0] if lrs is not None else args.features_lr
    lr2 = lrs[1] if lrs is not None else args.classifier_lr
    optimizer = optim.Adam(
        [
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if "fc" not in name
                ],
                "lr": lr1,
            },
            {"params": model.fc.parameters(), "lr": lr2},
        ],
        weight_decay=wd,
        betas=b,
    )
    return optimizer


class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class MLP(nn.Module):
    def __init__(self, hidden_size=100, num_layers=1):
        super(MLP, self).__init__()
        self.first_hidden_layer = HiddenLayer(1, hidden_size)
        self.rest_hidden_layers = nn.Sequential(
            *[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)
