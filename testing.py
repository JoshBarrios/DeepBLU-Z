from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path
import h5py
import numpy as np
from numpy import genfromtxt
import torchvision.transforms.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
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
# Creating data indices for training and validation splits:
dataset_size = len(ds)
indices = list(range(dataset_size))
split = int(np.floor(0.9999 * dataset_size))
batch_size = 4
# if 1:
#     np.random.seed(.5)
#     np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                                sampler=valid_sampler)


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
        y = transforms.functional.resize(x, [self.new_height, self.new_width])
        return y


def transform_input(im, pts, angle, new_height, new_width):
    h = im.shape[1]
    w = im.shape[2]
    num_pts = np.int(pts.shape[0] / 2)

    # # Choose random rotation angle and scaling for this batch
    # angle = random.choice(range(360))
    # scale = random.choice(np.linspace(0.2, 5, 49))
    # [new_height, new_width] = [np.int(np.round(h * scale)), np.int(np.round(w * scale))]
    warp_input = RotateAndScale(angle, new_height, new_width)

    transform_im = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Lambda(warp_input),
        transforms.ToTensor()
    ])
    # for the point image, we don't need to convert to PIL
    transform_ptim = transforms.Compose([
        transforms.Lambda(warp_input),
    ])

    # Get transformed point locations by creating an image of each point and performing the same transformation on it
    new_pts = np.zeros([16], dtype=int)
    for k in range(num_pts):
        pt_im = np.zeros([h, w], dtype='bool')
        pt_loc = [np.int(np.round(pts[k] * h)), np.int(np.round(pts[k + 8] * w))]
        # Single points often disappear when transformed, so we make a 4x4 point
        pt_im[np.int(pt_loc[0]) - 6:np.int(pt_loc[0]) + 6, np.int(pt_loc[1]) - 6:np.int(pt_loc[1]) + 6] = 1
        pt_im = Image.fromarray(pt_im)
        pt_im = transform_ptim(pt_im)
        pt_im = np.array(pt_im)
        pt = np.where(pt_im == 1)
        pt_inds = [np.int(np.round(np.mean(pt[0]))), np.int(np.round(np.mean(pt[1])))]
        new_pts[k] = pt_inds[0]
        new_pts[k + 8] = pt_inds[1]
    new_pts = torch.DoubleTensor(new_pts).unsqueeze(dim=0)

    new_im = transform_im(im).unsqueeze(dim=0)
    return new_im, new_pts


#%%
from torch import nn
import torchvision

class Model(nn.Module):
    def __init__(self,):
        super().__init__()

        backbone = 'resnet18'
        if backbone.startswith('mem-'):
            backbone = backbone[4:]

        # Implement backbones and change input and output sizes to meet our needs
        if backbone.startswith('resnet') or backbone.startswith('resnext'):
            first_conv = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
            self.features = getattr(torchvision.models, backbone)(pretrained=False)
            self.features.conv1 = first_conv
            self.features.fc = nn.Linear(self.features.fc.in_features, (8 * 2))

        elif backbone.startswith('efficientnet'):
            from efficientnet_pytorch import EfficientNet
            self.features = EfficientNet.from_name(backbone)
            self.features._conv_stem = nn.Conv2d(1, self.features._conv_stem.out_channels, kernel_size=3, stride=2,
                                                 padding=1,
                                                 bias=False)
            num_features = self.features._conv_head.out_channels
            self.features._fc = nn.Linear(num_features, (8 * 2), bias=False)
            self.features.set_swish(memory_efficient=True)

        elif backbone.startswith('densenet'):
            channels = 96 if backbone == 'densenet161' else 64
            first_conv = nn.Conv2d(1, channels, 7, 2, 3, bias=False)
            self.features = getattr(torchvision.models, backbone)(pretrained=False)
            self.features.features.conv0 = first_conv
            num_features = self.features.classifier.in_features
            self.features.classifier = nn.Linear(num_features, (8 * 2), bias=True)

        else:
            raise ValueError('wrong backbone')

    def forward(self, x):
        return self.features(x)

model = Model()
device = torch.device('cuda:0')
model.to(device)
# %%
import copy
device = torch.device('cuda:0')
for i, (images, targets) in enumerate(train_loader):
    # Choose random rotation angle and scaling for this batch
    angle = random.choice(range(360))
    scale = random.choice(np.linspace(0.2, 2, 25))
    scale = 2
    [new_height, new_width] = [np.int(np.round(images.size()[2] * scale)), np.int(np.round(images.size()[3] * scale))]
    new_ims, new_targets = transform_input(images[0], targets[0], angle, new_height, new_width)
    for l in range(len(images)):
        new_im, new_target = transform_input(images[l], targets[l], angle, new_height, new_width)
        new_ims = torch.cat((new_ims, new_im), dim=0)
        new_targets = torch.cat((new_targets, new_target), dim=0)

    # images = copy.deepcopy(new_ims)
    # targets = copy.deepcopy(new_targets)
    #
    # images = images.to(device)
    # targets = targets.to(device)
    #
    # output = model(images).to(torch.double)
