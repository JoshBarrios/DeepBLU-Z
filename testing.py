from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path
import h5py
import numpy as np
from numpy import genfromtxt
import torchvision.transforms.functional as F
import PIL
import random

# %%

normalize = transforms.Normalize(mean=[0.456],
                                 std=[0.224])


class ImagesDS(Dataset):

    def __init__(self, path):
        super().__init__()
        self.root = Path(path)

        # Get images from hdf file
        hdf_path = Path(self.root, 'images_ds.h5')
        hdf_file = h5py.File(hdf_path, 'r')
        for gname, group in hdf_file.items():
            for dname, ds in group.items():
                self.images = ds

        # Get tracking points from csv
        trck_pts = genfromtxt(Path(self.root, 'trck_pts.csv'), delimiter=',')
        trck_pts = np.transpose(trck_pts)
        trck_pts = trck_pts.reshape(np.int(trck_pts.shape[0] / 8), 8, 2)
        self.trck_pts = trck_pts
        self.num_pts = 8

    def __getitem__(self, index):

        # Get image from hdf5 dataset
        image = self.images[index, :, :]
        image = np.transpose(image)
        h = image.shape[0]
        w = image.shape[1]
        # Simulate RGB 3-channel image
        # image = np.repeat(image[..., np.newaxis], 3, -1)
        image = Image.fromarray(np.uint8(image))
        image = F.to_tensor(image)
        # image  = image.to(torch.double)
        image = normalize(image)

        # targets are the tracking point locations normalized to W and H of the image
        targets_ = self.trck_pts[index, :, :]
        targets = np.double(np.zeros(self.num_pts * 2))
        targets[0:self.num_pts] = targets_[:, 0] / h
        targets[self.num_pts:self.num_pts * 2] = targets_[:, 1] / w
        targets = torch.tensor(targets)

        return image, targets

    def __len__(self):
        return self.trck_pts.shape[0]


# %%

ds = ImagesDS(
    '/media/userman/249272E19272B6C0/Documents and Settings/jbarr/Documents/Douglass Lab/2020/2p behavior data/training_data/')

# %%

# define transform as class to allow identical transform on multiple images
class RotateAndScale:
    def __init__(self, angle, new_height, new_width):
        self.angle = angle
        self.new_height = new_height
        self.new_width = new_width

    def __call__(self, x):
        angle = self.angle
        x = transforms.functional.rotate(x, angle, expand=True)
        y = transforms.functional.resize(x, [new_height, new_width])
        return y

def transform_input(im, pts):
    h = im.shape[1]
    w = im.shape[2]
    num_pts = np.int(pts.shape[0] / 2)

    # Choose random rotation angle and scaling for this batch
    angle = random.choice(range(360))
    scale = random.choice(np.linspace(0.2, 5, 49))
    [new_height, new_width] = [np.int(np.round(h * scale)), np.int(np.round(w * scale))]
    warp_input = RotateAndScale(angle, new_height, new_width)

    transform = transforms.Compose([
        transforms.Lambda(warp_input)
    ])

    # Get transformed point locations by creating an image of each point and performing the same transformation on it
    new_pts = np.zeros([16], dtype=int)
    for k in range(num_pts):
        pt_im = np.zeros([h, w], dtype='bool')
        pt_loc = [np.int(np.round(pts[k] * h)), np.int(np.round(pts[k + 8] * w))]
        # Single points often disappear when transformed, so we make a 4x4 point
        pt_im[np.int(pt_loc[0]) - 4:np.int(pt_loc[0]) + 4, np.int(pt_loc[1]) - 4:np.int(pt_loc[1]) + 4] = 1
        pt_im = Image.fromarray(pt_im)
        pt_im = transform(pt_im)
        pt_im = np.array(pt_im)
        pt = np.where(pt_im == 1)
        pt_inds = [np.int(np.round(np.mean(pt[0]))), np.int(np.round(np.mean(pt[1])))]
        new_pts[k] = pt_inds[0]
        new_pts[k + 8] = pt_inds[1]

    new_im = transform(im)
    return new_im, new_pts