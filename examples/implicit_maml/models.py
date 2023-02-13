from collections import OrderedDict

import torch.nn as nn


def conv_block(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        OrderedDict(
            [
                ("conv", nn.Conv2d(in_channels, out_channels, **kwargs)),
                (
                    "norm",
                    nn.BatchNorm2d(
                        out_channels, momentum=1.0, track_running_stats=False
                    ),
                ),
                ("relu", nn.ReLU()),
                ("pool", nn.MaxPool2d(2)),
            ]
        )
    )


class ConvModel(nn.Module):
    """4-layer Convolutional Neural Network architecture from [1].
    Parameters
    ----------
    in_channels : int
        Number of channels for the input images.
    out_features : int
        Number of classes (output of the model).
    hidden_size : int (default: 64)
        Number of channels in the intermediate representations.
    feature_size : int (default: 64)
        Number of features returned by the convolutional head.
    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """

    def __init__(self, in_channels, out_features, hidden_size=64, feature_size=64):
        super(ConvModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.feature_size = feature_size

        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "layer1",
                        conv_block(
                            in_channels,
                            hidden_size,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (
                        "layer2",
                        conv_block(
                            hidden_size,
                            hidden_size,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (
                        "layer3",
                        conv_block(
                            hidden_size,
                            hidden_size,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (
                        "layer4",
                        conv_block(
                            hidden_size,
                            hidden_size,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                        ),
                    ),
                ]
            )
        )
        self.classifier = nn.Linear(feature_size, out_features, bias=True)

    def forward(self, inputs):
        features = self.features(inputs)
        features = features.view((features.size(0), -1))
        logits = self.classifier(features)
        return logits


def ConvOmniglot(out_features, hidden_size=64):
    return ConvModel(1, out_features, hidden_size=hidden_size, feature_size=hidden_size)


def ConvMiniImagenet(out_features, hidden_size=64):
    return ConvModel(
        3, out_features, hidden_size=hidden_size, feature_size=5 * 5 * hidden_size
    )
