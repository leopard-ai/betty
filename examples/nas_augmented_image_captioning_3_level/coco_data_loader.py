import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import numpy as np
import cv2


def collate_fn(data):
    data.sort(key=lambda x: x[2], reverse=True)

    images, captions, lengths, info = zip(*data)
    images = torch.stack(images, 0)
    # print(len(captions), max(lengths))
    targets = torch.zeros(len(captions), max(lengths)).long()

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, info


class CaptionDataset(Dataset):
    def __init__(self, json_file, h5py_file, args, transform=None):
        # self.split = split
        # assert self.split in {"train", "val", "test", "pseudo"}

        self.json_file = json.load(open(json_file, "r"))
        self.h5_file = None
        self.h5_path = h5py_file
        self.transform = transform
        self.args = args
        self.split_idxs = {}
        self.split = "train"
        for i, file in enumerate(self.json_file["images"]):
            if file["split"] not in self.split_idxs.keys():
                self.split_idxs[file["split"]] = []
            self.split_idxs[file["split"]].append(i)

    def __len__(self):
        return sum([len(v) for x, v in self.split_idxs.items()])

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        start = self.h5_file["label_start_ix"][idx] - 1
        end = self.h5_file["label_end_ix"][idx] + 1
        sample_seq = np.random.choice(np.arange(start, end))
        # print("sample_seq: ",sample_seq)
        # image_path = os.path.join(self.args.data,'coco2014','coco2014_imgs',
        # 							'img_'+str(idx)+'.png')
        # image = cv2.imread(image_path) # shape: (256,256,3)
        image = self.h5_file["images"][idx]
        # print(image.shape)
        # image = image.transpose(1,2,0)
        # print('img shape: ',image.shape)
        if self.transform is not None:
            image = self.transform(image)
            # print("image: ",image)
        else:
            image = transforms.ToTensor()(image).float()
        # print(image.shape) #shape: (3,256,256)

        label = self.h5_file["labels"][sample_seq]
        # print("label: ",label)
        length = self.h5_file["label_length"][sample_seq]
        # print("length: ",length)

        info = {}
        # print(self.split_idxs[self.split],idx)
        info["id"] = self.json_file["images"][idx]["id"]
        info["path"] = self.json_file["images"][idx]["file_path"]

        return image, torch.Tensor(label.astype("int")), length, info


def get_loader(
    json_file="data/cocotalk.json",
    h5_file="data/cocotalk.h5",
    transform=None,
    batch_size=5,
    shuffle=False,
    num_workers=1,
    args=None,
    debug=False,
):
    coco = CaptionDataset(
        json_file=json_file, h5py_file=h5_file, transform=transform, args=args
    )

    coco_json = json.load(open(json_file, "r"))
    indices = {"train": [], "val": [], "test": []}
    for i, file in enumerate(coco_json["images"]):
        indices[file["split"]].append(i)

    if debug:
        indices["train"] = np.array(indices["train"])[
            np.random.choice(len(indices["train"]), 10)
        ]
        indices["val"] = np.array(indices["val"])[
            np.random.choice(len(indices["val"]), 10)
        ]
        indices["test"] = np.array(indices["test"])[
            np.random.choice(len(indices["test"]), 10)
        ]
        print("size of train is now: ", len(indices["train"]))
    coco.split = "train"
    train_load = torch.utils.data.DataLoader(
        coco,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices["train"]),
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    coco.split = "val"
    val_load = torch.utils.data.DataLoader(
        coco,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices["val"]),
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    coco.split = "val"
    external_load = torch.utils.data.DataLoader(
        coco,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices["val"]),
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    return train_load, val_load, external_load


def get_pseudo_loader(model, input_external):
    # coco_json = json.load(open(json_file,'r'))
    # ix_2_word = coco_json['ix_to_word']
    features = model.encode(input_external)
    # print(features.shape)
    # features = Variable(torch.tensor(np.random.randn(80,512)),requires_grad=False).float().cuda()
    sampled_ids = model._decoder.sample(features.view(features.size(0), -1)).cpu()
    sampled_lens = []
    for i in range(sampled_ids.size(0)):
        col = np.where(np.array(sampled_ids[i].detach().cpu()) == 0)[0]
        if len(col) == 0:
            sampled_lens.append(sampled_ids.size(1))
        else:
            sampled_lens.append(col[0])

    input_external, sampled_ids, sampled_lens = zip(
        *sorted(
            zip(input_external, sampled_ids, sampled_lens),
            reverse=True,
            key=lambda x: x[2],
        )
    )

    input_external, sampled_ids = torch.stack(input_external, 0), torch.stack(
        sampled_ids, 0
    )
    sampled_lens = list(sampled_lens)
    # sampled_ids = pack_padded_sequence(sampled_ids, sampled_lens, batch_first=True)[0]
    return input_external.cuda(), sampled_ids.cuda(), sampled_lens
