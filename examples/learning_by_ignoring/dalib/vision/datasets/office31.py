from typing import Optional
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class Office31(ImageList):
    """Office31 Dataset.

    Parameters:
        - **root** (str): Root directory of dataset
        - **task** (str): The task (domain) to create dataset. Choices include ``'A'``: amazon, \
            ``'D'``: dslr and ``'W'``: webcam.
        - **download** (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        - **transform** (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, ``transforms.RandomCrop``.
        - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            amazon/
                images/
                    backpack/
                        *.jpg
                        ...
            dslr/
            webcam/
            image_list/
                amazon.txt
                dslr.txt
                webcam.txt
    """

    download_list = [
        (
            "image_list",
            "image_list.zip",
            "https://cloud.tsinghua.edu.cn/f/1f5646f39aeb4d7389b9/?dl=1",
        ),
        (
            "amazon",
            "amazon.tgz",
            "https://cloud.tsinghua.edu.cn/f/05640442cd904c39ad60/?dl=1",
        ),
        (
            "dslr",
            "dslr.tgz",
            "https://cloud.tsinghua.edu.cn/f/a069d889628d4b468c32/?dl=1",
        ),
        (
            "webcam",
            "amazon.tgz",
            "https://cloud.tsinghua.edu.cn/f/4c4afebf51384cf1aa95/?dl=1",
        ),
    ]
    image_list = {
        "A": "image_list/amazon.txt",
        "D": "image_list/dslr.txt",
        "W": "image_list/webcam.txt",
        "A_train": "image_list/amazon_train.txt",
        "A_val": "image_list/amazon_val.txt",
        "A_test": "image_list/amazon_test.txt",
        "D_train": "image_list/dslr_train.txt",
        "D_val": "image_list/dslr_val.txt",
        "D_test": "image_list/dslr_test.txt",
        "W_train": "image_list/webcam_train.txt",
        "W_val": "image_list/webcam_val.txt",
        "W_test": "image_list/webcam_test.txt",
        "A_train_big": "image_list_big/amazon_train.txt",
        "A_test_big": "image_list_big/amazon_test.txt",
        "D_train_big": "image_list_big/dslr_train.txt",
        "D_test_big": "image_list_big/dslr_test.txt",
        "W_train_big": "image_list_big/webcam_train.txt",
        "W_test_big": "image_list_big/webcam_test.txt",
        "AW_train": "image_list/amazon_web_train.txt",
        "WA_train": "image_list/amazon_web_train.txt",
        "AD_train": "image_list/amazon_dslr_train.txt",
        "DA_train": "image_list/amazon_dslr_train.txt",
        "WD_train": "image_list/webcam_dslr_train.txt",
        "DW_train": "image_list/webcam_dslr_train.txt",
    }
    CLASSES = [
        "back_pack",
        "bike",
        "bike_helmet",
        "bookcase",
        "bottle",
        "calculator",
        "desk_chair",
        "desk_lamp",
        "desktop_computer",
        "file_cabinet",
        "headphones",
        "keyboard",
        "laptop_computer",
        "letter_tray",
        "mobile_phone",
        "monitor",
        "mouse",
        "mug",
        "paper_notebook",
        "pen",
        "phone",
        "printer",
        "projector",
        "punchers",
        "ring_binder",
        "ruler",
        "scissors",
        "speaker",
        "stapler",
        "tape_dispenser",
        "trash_can",
    ]

    def __init__(self, root: str, task: str, download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])
        if task == "A_train":
            domain_idx = 0
        elif task == "D_train":
            domain_idx = 1
        elif task == "W_train":
            domain_idx = 2
        else:
            domain_idx = -1

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(
                map(
                    lambda file_name, _: check_exits(root, file_name),
                    self.download_list,
                )
            )

        super(Office31, self).__init__(
            root,
            Office31.CLASSES,
            data_list_file=data_list_file,
            domain_idx=domain_idx,
            **kwargs
        )
