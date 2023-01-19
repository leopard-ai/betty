import numpy as np
from PIL import Image
import h5py
import argparse

parser = argparse.ArgumentParser("converter")
parser.add_argument(
    "--inp_data",
    type=str,
    default="data/cocotalk.h5",
    help="location of the data hdf5 file",
)
parser.add_argument(
    "--out_data",
    type=str,
    default="data/coco2014/coco2014_imgs",
    help="location of the output folder",
)

args = parser.parse_args()
h5_file = h5py.File(args.inp_data, "r")

n = np.shape(h5_file["images"])[0]

for i in range(n):
    if (i + 1) % 50 == 0:
        print("{}/{} Done".format(i + 1, n + 1))
    arr = h5_file["images"][i].transpose(1, 2, 0)
    im = Image.fromarray(arr, "RGB")
    name = "img_{}.png".format(i)
    im.save(args.out_data + name)
print("all done!")
