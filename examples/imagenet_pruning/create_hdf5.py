"""Create ImageNet HDF5 from standard directory structure."""
import io
import os

import click
import h5py
import numpy as np
from pathlib import Path
from PIL import Image


IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif"]


def _find_classes(root_dir):
    """Finds the class folders in a dataset.
    Args:
        root_dir (string): Root directory path.
    Returns:
        tuple: (classes, class_to_idx) where classes are relative to
            (root_dir), and class_to_idx is a dictionary.
    Ensures:
        No class is a subdirectory of another.
    """
    classes = [d.name for d in os.scandir(root_dir) if d.is_dir()]
    classes.sort()

    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


def _has_file_allowed_extension(filename):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


@click.command()
@click.option("--out", type=str, default=None, help="Out filepath.")
@click.option("--root-dir", type=str, default=None, help="ImageNet directory.")
@click.option("--short-size", type=int, default=None, help="Downsample images to this.")
def create_hdf5(out, root_dir, short_size):
    root_dir = os.path.expanduser(root_dir)
    # root_dir = path.Path(root_dir)
    root_dir = Path(root_dir)

    out_h5file = h5py.File(out, "w")
    out_h5file.attrs["title"] = "ImageNet"

    special_uint8 = h5py.special_dtype(vlen=np.dtype("uint8"))
    grps = {
        "train": out_h5file.create_group("train"),
        "val": out_h5file.create_group("val"),
    }
    for target in ["train", "val"]:
        d = os.path.join(root_dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            if len(fnames) == 0:
                continue

            num_ex = len(os.listdir(root))
            print(root, num_ex)

            class_name = os.path.basename(root)
            grps[target].create_dataset(class_name, (num_ex,), dtype=special_uint8)
            assert len(fnames) == num_ex

            for ex_i, fname in enumerate(sorted(fnames)):
                if _has_file_allowed_extension(fname):
                    with open(os.path.join(root, fname), "rb") as f:
                        img_bytes = f.read()

                    img_io = io.BytesIO(img_bytes)
                    img = Image.open(img_io)
                    w, h = img.size

                    if (short_size is not None) and (min(w, h) > short_size):
                        img.load()

                        if h > w:
                            out_w = short_size
                            out_h = round((h * out_w) / w)
                        else:
                            out_h = short_size
                            out_w = round((w * out_h) / h)

                        img = img.resize((out_w, out_h), resample=Image.BILINEAR)

                        img_io = io.BytesIO()
                        img.save(img_io, format="jpeg")
                        img_io.seek(io.SEEK_SET)
                        img_bytes = img_io.read()

                    grps[target][class_name][ex_i] = np.frombuffer(
                        img_bytes, dtype=np.uint8
                    )
                else:
                    assert False, fname


if __name__ == "__main__":
    create_hdf5()
