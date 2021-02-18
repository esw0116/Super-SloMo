from model.extraction.common import pixel_reshuffle, TwoConvBlockIN, FourConvResBlockIN, FourDilateConvResBlockIN

import torch
from torch import nn


class Center(nn.Module):
    """
    Follow the same model in 'Learning to Extract a Video Sequence from a Single Motion-Blurred Image' center estimation
    """
    def __init__(self):
        super(Center, self).__init__()
        self.conv1_B  = nn.Conv2d( 16,  144, (5, 5), (1, 1), (2, 2), bias=False)
        self.conv2  = nn.Conv2d( 3,  64, (3, 3), (1, 1), (1, 1), bias=False)
        
        self.norm3 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv3  = nn.Conv2d( 64,  3, (3, 3), (1, 1), (1, 1), bias=False)

        self.pixel_shuffle = nn.PixelShuffle(4)
        
        self.LocalGrad1 = self.make_layer(FourConvResBlockIN, 144, 144, 3)
        self.LocalGrad2 = self.make_layer2(144, 144)
        self.LocalGrad3 = self.make_layer3(144, 144)
        self.LocalGrad4 = self.make_layer(FourConvResBlockIN, 144, 144, 3)

        self.fuse1 = self.make_layer(TwoConvBlockIN, 144, 16, 1)
        self.fuse2 = self.make_layer(TwoConvBlockIN, 144, 16, 1)
        self.fuse3 = self.make_layer(TwoConvBlockIN, 144, 16, 1)
        self.fuse4 = self.make_layer(TwoConvBlockIN, 144, 16, 1)

        self.GlobalGrad = self.make_layer(FourConvResBlockIN, 64, 64, 1)

    def forward(self, Blurry):
        Blurry_r = Blurry[:,0,:,:].unsqueeze(1)
        Blurry_g = Blurry[:,1,:,:].unsqueeze(1)
        Blurry_b = Blurry[:,2,:,:].unsqueeze(1)

        Blurry_r_0 = pixel_reshuffle(Blurry_r, 4)
        Blurry_g_0 = pixel_reshuffle(Blurry_g, 4)
        Blurry_b_0 = pixel_reshuffle(Blurry_b, 4)

        x_r_1 = self.conv1_B(Blurry_r_0)
        x_r_1 = self.LocalGrad1(x_r_1)
        x_r_2 = self.LocalGrad2(x_r_1)
        x_r_3 = self.LocalGrad3(x_r_2)
        x_r_4 = self.LocalGrad4(x_r_3)

        x_g_1 = self.conv1_B(Blurry_g_0)
        x_g_1 = self.LocalGrad1(x_g_1)
        x_g_2 = self.LocalGrad2(x_g_1)
        x_g_3 = self.LocalGrad3(x_g_2)
        x_g_4 = self.LocalGrad4(x_g_3)

        x_b_1 = self.conv1_B(Blurry_b_0)
        x_b_1 = self.LocalGrad1(x_b_1)
        x_b_2 = self.LocalGrad2(x_b_1)
        x_b_3 = self.LocalGrad3(x_b_2)
        x_b_4 = self.LocalGrad4(x_b_3)

        x_r_1 = self.fuse1(x_r_1)
        x_r_2 = self.fuse2(x_r_2)
        x_r_3 = self.fuse3(x_r_3)
        x_r_4 = self.fuse4(x_r_4)

        x_g_1 = self.fuse1(x_g_1)
        x_g_2 = self.fuse2(x_g_2)
        x_g_3 = self.fuse3(x_g_3)
        x_g_4 = self.fuse4(x_g_4)

        x_b_1 = self.fuse1(x_b_1)
        x_b_2 = self.fuse2(x_b_2)
        x_b_3 = self.fuse3(x_b_3)
        x_b_4 = self.fuse4(x_b_4)

        # we only estimate the residual respect to sharp reference image.
        x_r_5 = self.pixel_shuffle(x_r_1 + x_r_2 + x_r_3 + x_r_4 + Blurry_r_0) 
        x_g_5 = self.pixel_shuffle(x_g_1 + x_g_2 + x_g_3 + x_g_4 + Blurry_g_0) 
        x_b_5 = self.pixel_shuffle(x_b_1 + x_b_2 + x_b_3 + x_b_4 + Blurry_b_0) 

        out = torch.cat( (x_r_5, x_g_5, x_b_5), 1)
        out = self.conv2(out)
        out = self.GlobalGrad(out)
        out = self.norm3(out)
        out = self.relu(out)
        out = self.conv3(out) + Blurry

        return out

    def make_layer(self, block, in_channels, out_channels, blocks):
        layers = []
        for i in range(1, blocks + 1):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def make_layer2(self, in_channels, out_channels):
        layers = []
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 1, 2))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 2, 4))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 4, 8))
        return nn.Sequential(*layers)

    def make_layer3(self, in_channels, out_channels):
        layers = []
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 8, 4))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 4, 2))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 2, 1))
        return nn.Sequential(*layers)
