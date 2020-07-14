'''
Josh Barrios 7/9/2020
Defining behavior video dataset with tracking points.
Defining training and validation dataloaders.

'''
# %%
import numpy as np
from numpy import genfromtxt
from pathlib import Path
import h5py
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import torchvision.transforms.functional as F

# %% Define dataset
# define normalization function for prepping input to resnet18


normalize = transforms.Normalize(mean=[0.456],
                                 std=[0.224])


class ImagesDS(Dataset):

    def __init__(self, args):
        super().__init__()
        self.root = Path(args.datapath)

        # Get images from hdf file
        hdf_path = Path(self.root, 'images_ds.h5')
        hdf_file = h5py.File(hdf_path, 'r')
        for gname, group in hdf_file.items():
            for dname, ds in group.items():
                self.images = ds

        # Get tracking points from csv
        trck_pts = genfromtxt(Path(self.root, 'trck_pts.csv'), delimiter=',')
        trck_pts = np.transpose(trck_pts)
        trck_pts = trck_pts.reshape(np.int(trck_pts.shape[0] / args.num_pts), args.num_pts, 2)
        self.trck_pts = trck_pts
        self.num_pts = args.num_pts

    def __getitem__(self, index):

        # Get image from hdf5 dataset
        image = self.images[index, :, :]
        image = np.transpose(image)
        h = image.shape[0]
        w = image.shape[1]
        # Simulate RGB 3-channel image
        #image = np.repeat(image[..., np.newaxis], 3, -1)
        image = Image.fromarray(np.uint8(image))
        image = F.to_tensor(image)
        # image  = image.to(torch.double)
        image = normalize(image)

        # targets are the tracking point locations normalized to W and H of the image
        targets_ = self.trck_pts[index, :, :]
        targets = np.double(np.zeros(self.num_pts*2))
        targets[0:self.num_pts] = targets_[:, 0] / h
        targets[self.num_pts:self.num_pts*2] = targets_[:, 1] / w
        targets = torch.tensor(targets)

        return image, targets

    def __len__(self):
        return self.trck_pts.shape[0]


# %% Define training and validation loaders

def get_train_val_loader(args):
    images_ds = ImagesDS(args)

    # Creating data indices for training and validation splits:
    dataset_size = len(images_ds)
    indices = list(range(dataset_size))
    split = int(np.floor(args.val_split * dataset_size))
    if args.shuffle:
        np.random.seed(args.seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(images_ds, batch_size=args.batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(images_ds, batch_size=args.batch_size,
                                                    sampler=valid_sampler)

    return train_loader, validation_loader