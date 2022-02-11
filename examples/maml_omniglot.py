import os
import os.path
import errno
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
from PIL import Image

from betty.module import Module, HypergradientConfig
from betty.engine import Engine


argparser = argparse.ArgumentParser()
argparser.add_argument('--n_way', type=int, help='n way', default=5)
argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
argparser.add_argument('--inner_steps', type=int, help='number of inner steps', default=5)
argparser.add_argument('--device', type=str, help='device', default='cuda')
argparser.add_argument('--task_num',type=int, help='meta batch size, namely task num', default=5)
argparser.add_argument('--seed', type=int, help='random seed', default=1)
arg = argparser.parse_args()

torch.manual_seed(arg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(arg.seed)
np.random.seed(arg.seed)

class Omniglot(data.Dataset):
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    '''
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    '''

    def __init__(self, root, transform=None, target_transform=None,
                 download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        self.all_items = find_classes(os.path.join(self.root, self.processed_folder))
        self.idx_classes = index_classes(self.all_items)

    def __getitem__(self, index):
        filename = self.all_items[index][0]
        img = str.join('/', [self.all_items[index][2], filename])

        target = self.idx_classes[self.all_items[index][1]]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.all_items)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, "images_evaluation")) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, "images_background"))

    def download(self):
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            file_processed = os.path.join(self.root, self.processed_folder)
            print("== Unzip from " + file_path + " to " + file_processed)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(file_processed)
            zip_ref.close()
        print("Download finished.")


def find_classes(root_dir):
    retour = []
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            if (f.endswith("png")):
                r = root.split('/')
                lr = len(r)
                retour.append((f, r[lr - 2] + "/" + r[lr - 1], root))
    print("== Found %d items " % len(retour))
    return retour


def index_classes(items):
    idx = {}
    for i in items:
        if i[1] not in idx:
            idx[i[1]] = len(idx)
    print("== Found %d classes" % len(idx))
    return idx


class OmniglotNShot:

    def __init__(self, root, batchsz, n_way, k_shot, k_query, imgsz, device=None):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """

        self.resize = imgsz
        self.device = device
        if not os.path.isfile(os.path.join(root, 'omniglot.npy')):
            # if root/data.npy does not exist, just download it
            self.x = Omniglot(
                root, download=True,
                transform=transforms.Compose(
                    [lambda x: Image.open(x).convert('L'),
                     lambda x: x.resize((imgsz, imgsz)),
                     lambda x: np.reshape(x, (imgsz, imgsz, 1)),
                     lambda x: np.transpose(x, [2, 0, 1]),
                     lambda x: x / 255.]),
            )

            temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
            for (img, label) in self.x:
                if label in temp.keys():
                    temp[label].append(img)
                else:
                    temp[label] = [img]

            self.x = []
            for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
                self.x.append(np.array(imgs))

            # as different class may have different number of imgs
            self.x = np.array(self.x).astype(np.float)  # [[20 imgs],..., 1623 classes in total]
            # each character contains 20 imgs
            print('data shape:', self.x.shape)  # [1623, 20, 84, 84, 1]
            temp = []  # Free memory
            # save all dataset into npy file.
            np.save(os.path.join(root, 'omniglot.npy'), self.x)
            print('write into omniglot.npy.')
        else:
            # if data.npy exists, just load it.
            self.x = np.load(os.path.join(root, 'omniglot.npy'))
            print('load from omniglot.npy.')

        # [1623, 20, 84, 84, 1]
        # TODO: can not shuffle here, we must keep training and test set distinct!
        self.x_train, self.x_test = self.x[:1200], self.x[1200:]

        # self.normalization()

        self.batchsz = batchsz
        self.n_cls = self.x.shape[0]  # 1623
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        assert (k_shot + k_query) <= 20

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        print("DB: train", self.x_train.shape, "test", self.x_test.shape)

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),  # current epoch data cached
                               "test": self.load_data_cache(self.datasets["test"])}

    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        # print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)

    # print("after norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)

    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)

                for j, cur_class in enumerate(selected_cls):

                    selected_img = np.random.choice(20, self.k_shot + self.k_query, False)

                    # meta-training and meta-test
                    x_spt.append(data_pack[cur_class][selected_img[:self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.reshape(np.array(x_spt), (self.n_way * self.k_shot, 1, self.resize, self.resize))[perm]
                y_spt = np.reshape(np.array(y_spt), (self.n_way * self.k_shot))[perm]
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.reshape(np.array(x_qry), (self.n_way * self.k_query, 1, self.resize, self.resize))[perm]
                y_qry = np.reshape(np.array(y_qry), (self.n_way * self.k_query))[perm]

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)

            # [b, setsz, 1, 84, 84]
            x_spts = np.reshape(np.array(x_spts).astype(np.float32), (self.batchsz, setsz, 1, self.resize, self.resize))
            y_spts = np.reshape(np.array(y_spts).astype(np.int), (self.batchsz, setsz))
            # [b, qrysz, 1, 84, 84]
            x_qrys = np.reshape(np.array(x_qrys).astype(np.float32), (self.batchsz, querysz, 1, self.resize, self.resize))
            y_qrys = np.reshape(np.array(y_qrys).astype(np.int), (self.batchsz, querysz))

            x_spts, y_spts, x_qrys, y_qrys = [
                torch.from_numpy(z).to(self.device) for z in
                [x_spts, y_spts, x_qrys, y_qrys]
            ]

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch

db = OmniglotNShot(
        '/tmp/omniglot-data',
        batchsz=arg.task_num,
        n_way=arg.n_way,
        k_shot=arg.k_spt,
        k_query=arg.k_qry,
        imgsz=28,
        device=arg.device,
    )

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Net(nn.Module):
    def __init__(self, n_way, device):
        super(Net, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 64, 3),
                                 nn.BatchNorm2d(64, momentum=1, affine=True),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2, 2),
                                 nn.Conv2d(64, 64, 3),
                                 nn.BatchNorm2d(64, momentum=1, affine=True),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2, 2),
                                 nn.Conv2d(64, 64, 3),
                                 nn.BatchNorm2d(64, momentum=1, affine=True),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2, 2),
                                 Flatten(),
                                 nn.Linear(64, n_way)).to(device)

    def forward(self, x):
        return self.net.forward(x)


class Parent(Module):
    def forward(self, *args, **kwargs):
        return self.params, self.buffers

    def calculate_loss(self, batch, *args, **kwargs):
        inputs, targets = batch
        loss = 0
        for child in self._children:
            out = child(inputs)
            loss += F.cross_entropy(out, targets)

        return loss

    def configure_data_loader(self):
        return None

    def configure_module(self):
        return Net(arg.n_way, self.device)

    def configure_optimizer(self):
        return optim.Adam(self.module.parameters(), lr=0.01)


class Child(Module):
    def forward(self, x):
        return self.fmodule(self.params, self.buffers, x)

    def calculate_loss(self, batch, *args, **kwargs):
        if self.count % arg.inner_steps == 1:
            self.initialize_params()
        inputs, targets = batch
        out = self.fmodule(self.params, self.buffers, inputs)
        loss = F.cross_entropy(out, targets)

        return loss

    def initialize_params(self):
        assert len(self._parents) == 1
        p = self._parents[0]
        params, buffers = p()
        self.params, self.buffers = params.clone(), buffers.clone()

    def configure_data_loader(self):
        return None

    def configure_module(self):
        return Net(arg.n_way, self.device)

    def configure_optimizer(self):
        return optim.Adam(self.module.parameters(), lr=0.01)

parent_config = HypergradientConfig(type='maml',
                                    step=arg.inner_step,
                                    first_order=False,
                                    leaf=False)
child_config = HypergradientConfig(type='maml',
                                   step=1,
                                   first_order=False,
                                   leaf=True)

parent = Parent(config=parent_config, device=arg.device)
children = [Child(config=child_config, device=arg.device) for _ in range(5)]
problems = children + [parent]
dependencies = {parent: children}
engine = Engine(config=None, problems=problems, dependencies=dependencies)
engine.train()