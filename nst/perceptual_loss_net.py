from collections import namedtuple

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

# Define which layers to use
CONTENT_LAYER = 'relu2_2'
STYLE_LAYERS = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']


class PerceptualLossNet(nn.Module):
    """
    Perceptual loss network leveraging pretrained VGG-16
    """
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).eval()
        vgg_features = vgg.features
        self.layer_names = STYLE_LAYERS

        # Slices for each block/layer
        self.slice1 = nn.Sequential(*vgg_features[:4])   # relu1_2
        self.slice2 = nn.Sequential(*vgg_features[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*vgg_features[9:16]) # relu3_3
        self.slice4 = nn.Sequential(*vgg_features[16:23])# relu4_3

        # Freeze VGG params
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        relu1_2 = h
        h = self.slice2(h)
        relu2_2 = h
        h = self.slice3(h)
        relu3_3 = h
        h = self.slice4(h)
        relu4_3 = h
        vgg_output = namedtuple("VGGOutput", self.layer_names)
        return vgg_output(relu1_2, relu2_2, relu3_3, relu4_3)

def gram_matrix(feat):
    b, c, h, w = feat.size()
    f = feat.view(b, c, h * w)
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)

def total_variation_loss(img):
    b, c, h, w = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (b * c * h * w)
