from model.extraction.common import pixel_reshuffle, TwoConvBlockIN, FourConvResBlockINDis

import torch
from torch import nn
import pytorch_lightning as pl

class Dis_Center0(pl.LightningModule):
    """
    Follow the same model in 'Learning to Extract a Video Sequence from a Single Motion-Blurred Image' center estimation
    """
    def __init__(self, *args, **kwargs):
        super(Dis_Center0, self).__init__()
        self.conv1_B  = nn.Conv2d(48,  144, (5, 5), (1, 1), (2, 2), bias=False)
        self.conv2  = nn.Conv2d(144,  64, (3, 3), (1, 1), (1, 1), bias=False)
        
        self.norm3 = nn.InstanceNorm2d(64, affine=True)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv3  = nn.Conv2d( 64,  3, (3, 3), (1, 1), (1, 1), bias=False)

        self.pixel_shuffle = nn.PixelShuffle(4)
        self.pool = nn.MaxPool2d(2, stride=2)
        
        self.LocalGrad1 = self.make_layer(FourConvResBlockINDis, 144, 144, 3)
        self.LocalGrad2 = self.make_layer(FourConvResBlockINDis, 144, 144, 3)

        self.GlobalGrad = self.make_layer(FourConvResBlockINDis, 64, 64, 1)

    def forward(self, img1):
        img1_0 = pixel_reshuffle(img1, 4)

        img1_1 = self.conv1_B(img1_0)
        img1_1 = self.LocalGrad1(img1_1)
        img1_1 = self.pool(img1_1)
        img1_2 = self.LocalGrad2(img1_1)
        img1_2 = self.pool(img1_2)
        out = self.conv2(img1_2)
        out = self.GlobalGrad(out)
        out = self.norm3(out)
        out = self.lrelu(out)
        out = self.conv3(out)

        return out

    def make_layer(self, block, in_channels, out_channels, blocks):
        layers = []
        for i in range(1, blocks + 1):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)
