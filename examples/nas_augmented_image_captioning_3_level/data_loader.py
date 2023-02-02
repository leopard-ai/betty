import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from pycocotools.coco import COCO
from build_vocab import *
import sys
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

sys.path.append(".")
import pickle

nltk.download("punkt")


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(
        self,
        args=None,
        root="./data/train2014",
        json="./data/annotations/captions_train2014.json",
        vocab="./data/vocab.pkl",
        transform=None,
    ):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        print(json)
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        self.args = args

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]["caption"]
        img_id = coco.anns[ann_id]["image_id"]
        path = coco.loadImgs(img_id)[0]["file_name"]

        image = Image.open(os.path.join(self.root, path)).convert(
            "RGB"
        )  # burnout in colab
        # image = Image.open('./data/val2014/000000025394.jpg').convert('RGB') # using dummy as filler
        # image = Image.fromarray(np.uint8(np.random.randn(224,224,3)*100)) # dummy

        if self.transform is not None:
            image = self.transform(image)
            # print("image: ",image)
        else:
            image = transforms.ToTensor()(image).float()

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab("<start>"))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab("<end>"))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    # targets = pack_padded_sequence(targets, lengths, batch_first=True)[0]
    # print(targets,targets.shape)
    return images, targets, lengths


def get_loader(
    root="./data/train",
    json="./data/annotations/captions_train2014.json",
    vocab=None,
    transform=None,
    batch_size=5,
    shuffle=False,
    num_workers=1,
    args=None,
):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""

    assert vocab != None, "please provide Vocabulary."
    # COCO caption dataset
    coco = CocoDataset(
        root=root, json=json, vocab=vocab, transform=transform, args=args
    )
    # print('Total number of training points: ',len(coco))
    print("Vocab size: ", len(vocab))
    num_train = len(coco)
    indices = list(range(num_train))
    split = int(np.floor(coco.args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        coco,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    valid_queue = torch.utils.data.DataLoader(
        coco,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    external_queue = torch.utils.data.DataLoader(
        coco,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # the dataset for data selection. can be imagenet or so.
    # external_queue = torch.utils.data.DataLoader(
    #   coco, batch_size=self.args.batch_size,
    #   sampler=torch.utils.data.sampler.SubsetRandomSampler(
    #       indices[split:num_train]),
    # pin_memory=False, num_workers=4, collate_fn=collate_fn)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    return train_queue, valid_queue, external_queue


def get_pseudo_loader(model, input_external, vocab):
    features = model.encode(input_external)
    # print(features.shape)
    # features = Variable(torch.tensor(np.random.randn(80,512)),requires_grad=False).float().cuda()
    sampled_ids = model._decoder.sample(features.view(features.size(0), -1)).cpu()
    sampled_lens = []
    for i in range(sampled_ids.size(0)):
        col = np.where(sampled_ids[i] == vocab("<end>"))[0]
        if len(col) == 0:
            sampled_lens.append(sampled_ids.size(1))
        else:
            sampled_lens.append(col[0] + 1)
    # sampled_ids = Variable(sampled_ids, requires_grad=False)
    # data = zip(input_external, sampled_ids, sampled_lens)
    # data.sort(reverse=True, key= lambda x: x[2])
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
