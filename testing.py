import torchvision
from torch import nn
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

device = torch.device("cuda:0")

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        first_conv = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.features = getattr(torchvision.models, 'resnet18')(pretrained=False)
        self.features.conv1 = first_conv
        self.features.fc = nn.Linear(self.features.fc.in_features, (8 * 2))

    def forward(self, x):
        return self.features(x)

model = Model()
model.load_state_dict(torch.load('./models/090920/resnet18_50e2x/resnet18'))
model.to(device)

# Import database of images
import h5py
hdf_path = './data/training_data/images_ds.h5'
# hdf_path = '/home/userman/Documents/aws/danionella_training_data/images_ds.h5'
hdf_file = h5py.File(hdf_path, 'r')
for gname, group in hdf_file.items():
    for dname, ds in group.items():
        images = ds

from numpy import genfromtxt
# Import tracking points
ref_pts = genfromtxt('./data/training_data/trck_pts.csv', delimiter=',')
ref_pts = np.transpose(ref_pts)
ref_pts = ref_pts.reshape(np.int(ref_pts.shape[0] / 8), 8, 2)

imnum = 22400

refpoints = ref_pts[imnum, :, :]

image = images[imnum, :, :]
image = np.transpose(image)

h = image.shape[0]
w = image.shape[1]

im1 = Image.fromarray(image)
im = im1.convert('RGB')
normalize = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
im = normalize(im)

im.to(device)
im = im.type(torch.cuda.FloatTensor)
im.type()

im = torch.unsqueeze(im, 0)


output = model(im)
output = output.cpu()
output = output.detach().numpy()
output = np.squeeze(output)

trck_pts = np.zeros([2, 8])
trck_pts[0, :] = output[0:8] * h
trck_pts[1, :] = output[8:16] * w
trck_pts[trck_pts < 0] = 0
trck_pts = np.round(trck_pts)
trck_pts = np.transpose(trck_pts)


import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()
#%%
# Mark tracking points in image
image_marked = image
for pt in trck_pts:
    pt = np.uint16(pt)
    print(pt)
    image_marked[pt[0] - 4:pt[0] + 4, pt[1] - 4:pt[1] + 4] = 255
im = Image.fromarray(image_marked)
im.show()

#%%
print(output[0:7])