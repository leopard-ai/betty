import torch
import numpy as np
import torchvision.transforms as transforms


def transform(train=True):
    OFFICE_MEAN = [0.485, 0.456, 0.406]
    OFFICE_STD = [0.229, 0.224, 0.225]
    resize = 256
    randomResizedCrop = 224
    if train:
        data_transforms = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.RandomResizedCrop(randomResizedCrop),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(OFFICE_MEAN, OFFICE_STD),
            ]
        )
    else:
        data_transforms = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(randomResizedCrop),
                transforms.ToTensor(),
                transforms.Normalize(OFFICE_MEAN, OFFICE_STD),
            ]
        )
    return data_transforms
