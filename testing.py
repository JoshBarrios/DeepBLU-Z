import torch
from torch import nn
import torchvision

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        first_conv = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.features = getattr(torchvision.models, 'resnet18')(pretrained=False)
        self.features.conv1 = first_conv
        self.features.fc = nn.Linear(self.features.fc.in_features, (8 * 2))

model = Model()
model = model.load_state_dict(torch.load('/home/userman/PycharmProjects/DeepBLU-Z/models/082820/resnet18_50epochs'))


