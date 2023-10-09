import h5py
import io
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode


class ImageNet(Dataset):
    def __init__(
        self,
        dataset_file: str,
        sample_set: str,
        classes: list,
        transform_type: str,
        args,
    ):
        self.dataset = dataset_file
        self.sample_set = sample_set
        self.classes = classes
        self.class_map = {c: i for i, c in enumerate(classes)}

        # These are known ImageNet values
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        interpolation = InterpolationMode(args.interpolation)
        if transform_type == "train":
            # self.resizedcrop = transforms.RandomResizedCrop(224)
            # transforms.ColorJitter(brightness=0.4, contrast=0.4,
            #                        saturation=0.4, hue=0.4),
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        args.train_crop_size, interpolation=interpolation
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            # self.resizedcrop = transforms.Compose(
            #     [transforms.Resize(256), transforms.CenterCrop(224)]
            # )
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        args.val_resize_size, interpolation=interpolation
                    ),
                    transforms.CenterCrop(args.val_crop_size),
                    transforms.PILToTensor(),
                    transforms.ConvertImageDtype(torch.float),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

        self.sample_ids = []
        with h5py.File(self.dataset, "r") as dataset:
            for class_ in self.classes:
                for i in range(int(dataset[f"{sample_set}"][f"{class_}"].shape[0])):
                    self.sample_ids.append((class_, i))

    def __getitem__(self, i):
        class_, id_ = self.sample_ids[i]
        with h5py.File(self.dataset, "r") as dataset:
            image = dataset[f"{self.sample_set}"][f"{class_}"][id_]

        image = Image.open(io.BytesIO(image))
        if image.mode != "RGB":
            image = image.convert("RGB")
        # image = self.resizedcrop(image)
        image = self.transform(image)
        # image = np.array(image)

        return image, self.class_map[class_]

    def __len__(self):
        return len(self.sample_ids)


def get_subset_data(
    dataset: Dataset,
    prune_strategy: str,
    instance_weights_dir: str,
    frac_data_kept: float,
):
    if prune_strategy == "metaweight":
        print("Metaweight pruning strategy!")
        print("Load : ", os.path.join(instance_weights_dir, "sorted_idx.pt"))
        sorted_idx = torch.load(os.path.join(instance_weights_dir, "sorted_idx.pt"))
    else:
        print("Random pruning strategy!")
        print("Load : ", os.path.join(instance_weights_dir, "sorted_idx.pt"))
        # sorted_idx = np.random.permutation(len(dataset))
        sorted_idx = torch.load(os.path.join(instance_weights_dir, "sorted_idx.pt"))
        print(sorted_idx[:20])
    # sorted_weight = torch.load(os.path.join(instance_weights_dir, "sorted_weight.pt"))
    # weights_total = torch.load(os.path.join(instance_weights_dir, "weights_total.pt"))

    num_examples = len(dataset)
    print("No. of total examples (without pruning): ", num_examples)
    num_examples_data_kept = int(num_examples * frac_data_kept)
    print("No. of examples (after pruning): ", num_examples_data_kept)
    print("Fraction of data kept: ", frac_data_kept)

    selected_indices = sorted_idx[:num_examples_data_kept]

    return torch.utils.data.Subset(dataset=dataset, indices=selected_indices)
