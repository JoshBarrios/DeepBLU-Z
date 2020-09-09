'''
Josh Barrios 7/9/2020
Defining model

'''
# %%
import torchvision
from torch import nn


# %%

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        backbone = args.backbone
        if args.backbone.startswith('mem-'):
            backbone = args.backbone[4:]

        # Implement backbones and change input and output sizes to meet our needs
        if backbone.startswith('resnet') or backbone.startswith('resnext'):
            first_conv = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            self.features = getattr(torchvision.models, backbone)(pretrained=False)
            self.features.conv1 = first_conv
            self.features.fc = nn.Linear(self.features.fc.in_features, (args.num_pts * 2))

        elif backbone.startswith('efficientnet'):
            from efficientnet_pytorch import EfficientNet
            self.features = EfficientNet.from_name(backbone)
            self.features._conv_stem = nn.Conv2d(3, self.features._conv_stem.out_channels, kernel_size=3, stride=2,
                                                 padding=1,
                                                 bias=False)
            num_features = self.features._conv_head.out_channels
            self.features._fc = nn.Linear(num_features, (args.num_pts * 2), bias=False)
            self.features.set_swish(memory_efficient=True)

        elif backbone.startswith('densenet'):
            channels = 96 if backbone == 'densenet161' else 64
            first_conv = nn.Conv2d(3, channels, 7, 2, 3, bias=False)
            self.features = getattr(torchvision.models, backbone)(pretrained=False)
            self.features.features.conv0 = first_conv
            num_features = self.features.classifier.in_features
            self.features.classifier = nn.Linear(num_features, (args.num_pts * 2), bias=True)

        else:
            raise ValueError('wrong backbone')

    def forward(self, x):
        return self.features(x)
