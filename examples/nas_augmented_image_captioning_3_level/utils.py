import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import json


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    # print("output: ",output.shape)
    # print("target: ",target.shape)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print("pred: ",pred.shape, " target: ",target.shape)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print(correct)
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_coco(args):
    DEF_MEAN = [0.485, 0.456, 0.406]
    DEF_STD = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
            transforms.ToTensor(),
            transforms.Resize(112),  # smaller edge of image resized to 256
            # transforms.RandomCrop(192),                      # get 224x224 crop from random location
            transforms.Normalize(DEF_MEAN, DEF_STD),
        ]
    )

    return transform_train


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, valid_transform


def _data_transforms_cifar100(args):
    CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR_STD = [0.2675, 0.2565, 0.2761]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return (
        np.sum(
            np.prod(v.size())
            for name, v in model.named_parameters()
            if "auxiliary" not in name
        )
        / 1e6
    )


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, "checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, "model_best.pth.tar")
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = Variable(
            torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
        )
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print("Experiment dir : {}".format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, "scripts"))
        for script in scripts_to_save:
            dst_file = os.path.join(path, "scripts", os.path.basename(script))
            shutil.copyfile(script, dst_file)


# https://github.com/karpathy/neuraltalk2/blob/bd8c9d879f957e1218a8f9e1f9b663ac70375866/coco-caption/myeval.py
def score_lang(
    preds, args, annfile_path="annotations/captions_val2014.json", cider_only=True
):
    ckpt = {"val_predictions": preds}
    json.dump(ckpt, open("coco-caption/ckpt.json", "w"))
    os.chdir("coco-caption")
    if cider_only:
        os.system("python myCocoEvalCider.py ckpt")
    else:
        os.system("python myCocoEval.py ckpt")
    os.chdir("..")
    # coco = COCO(annfile_path)
    # valids = coco.getImgIds()

    # new_preds = [p for p in preds if preds['image_id'] in valids]
    # print('using %d/%d predictions' % (len(new_preds), len(preds)))

    # json.dump(new_preds, open('tmp.json', 'w'))
    # resFile = 'tmp.json'
    # cocoRes = coco.loadRes(resFile)
    # cocoEval = COCOEvalCap(coco, cocoRes)
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # cocoEval.evaluate()
    out = json.load(open("coco-caption/ckpt_out.json", "r"))
    # out = {}
    # for metric, score in cocoEval.eval.items():
    # 	out[metric] = score

    return out
