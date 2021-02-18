from model.extraction.common import pixel_reshuffle, TwoConvBlockIN, FourConvResBlockIN, FourDilateConvResBlockIN
from copy import deepcopy

import torch
from torch import nn


class N8_IN(nn.Module):
    def __init__(self):
        super(N8_IN, self).__init__()
        self.conv1_B  = nn.Conv2d(  16,  64, (5, 5), (1, 1), (2, 2), bias=False)
        self.conv1_S1  = nn.Conv2d( 16,  32, (5, 5), (1, 1), (2, 2), bias=False)
        self.conv1_S2  = nn.Conv2d( 16,  32, (5, 5), (1, 1), (2, 2), bias=False)
        self.conv2  = nn.Conv2d( 3,  32, (3, 3), (1, 1), (1, 1), bias=False)
        
        self.norm3 = nn.InstanceNorm2d(32, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv3  = nn.Conv2d( 32,  3, (3, 3), (1, 1), (1, 1), bias=False)

        self.pixel_shuffle = nn.PixelShuffle(4)
        
        self.LocalGrad1 = self.make_layer(FourConvResBlockIN, 128, 128, 4)
        self.LocalGrad2 = self.make_layer2(128, 128)
        self.LocalGrad3 = self.make_layer(FourConvResBlockIN, 128, 128, 4)

        self.fuse1 = self.make_layer(TwoConvBlockIN, 128, 16, 1)
        self.fuse2 = self.make_layer(TwoConvBlockIN, 128, 16, 1)
        self.fuse3 = self.make_layer(TwoConvBlockIN, 128, 16, 1)

        self.GlobalGrad = self.make_layer(TwoConvBlockIN, 32, 32, 1)

    def preprocess(self, img):
        img_r = img[:,0,:,:].unsqueeze(1)
        img_g = img[:,1,:,:].unsqueeze(1)
        img_b = img[:,2,:,:].unsqueeze(1)
        img_r_0 = pixel_reshuffle(img_r, 4)
        img_g_0 = pixel_reshuffle(img_g, 4)
        img_b_0 = pixel_reshuffle(img_b, 4)

        return img_r_0, img_g_0, img_b_0

    def forward(self, x1, x2, x3):
        x1_r_0, x1_g_0, x1_b_0 = self.preprocess(x1)
        x2_r_0, x2_g_0, x2_b_0 = self.preprocess(x2)
        x3_r_0, x3_g_0, x3_b_0 = self.preprocess(x3)

        x1_r_1 = self.conv1_B(x1_r_0)
        x2_r_1 = self.conv1_S1(x2_r_0)
        x3_r_1 = self.conv1_S2(x3_r_0)
        x_r_1 = torch.cat( (x1_r_1, x2_r_1, x3_r_1), 1 )

        x_r_1 = self.LocalGrad1(x_r_1)
        x_r_2 = self.LocalGrad2(x_r_1)
        x_r_3 = self.LocalGrad3(x_r_2)

        x1_g_1 = self.conv1_B(x1_g_0)
        x2_g_1 = self.conv1_S1(x2_g_0)
        x3_g_1 = self.conv1_S2(x3_g_0)
        x_g_1 = torch.cat( (x1_g_1, x2_g_1, x3_g_1), 1 )

        x_g_1 = self.LocalGrad1(x_g_1)
        x_g_2 = self.LocalGrad2(x_g_1)
        x_g_3 = self.LocalGrad3(x_g_2)

        x1_b_1 = self.conv1_B(x1_b_0)
        x2_b_1 = self.conv1_S1(x2_b_0)
        x3_b_1 = self.conv1_S2(x3_b_0)
        x_b_1 = torch.cat( (x1_b_1, x2_b_1, x3_b_1), 1 )

        x_b_1 = self.LocalGrad1(x_b_1)
        x_b_2 = self.LocalGrad2(x_b_1)
        x_b_3 = self.LocalGrad3(x_b_2)

        x_r_1 = self.fuse1(x_r_1)
        x_r_2 = self.fuse2(x_r_2)
        x_r_3 = self.fuse3(x_r_3)

        x_g_1 = self.fuse1(x_g_1)
        x_g_2 = self.fuse2(x_g_2)
        x_g_3 = self.fuse3(x_g_3)

        x_b_1 = self.fuse1(x_b_1)
        x_b_2 = self.fuse2(x_b_2)
        x_b_3 = self.fuse3(x_b_3)

        # we only estimate the residual respect to sharp reference image.
        x_r_5 = self.pixel_shuffle(x_r_1 + x_r_2 + x_r_3  + x3_r_0) 
        x_g_5 = self.pixel_shuffle(x_g_1 + x_g_2 + x_g_3  + x3_g_0) 
        x_b_5 = self.pixel_shuffle(x_b_1 + x_b_2 + x_b_3  + x3_b_0) 

        out = torch.cat( (x_r_5, x_g_5, x_b_5), 1)

        out = self.conv2(out)
        out = self.GlobalGrad(out)
        out = self.norm3(out)
        out = self.relu(out)
        out = self.conv3(out)
        return out

    def make_layer(self, block, in_channels, out_channels, blocks):
        layers = []
        for i in range(1, blocks + 1):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def make_layer2(self, in_channels, out_channels):
        layers = []
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 1, 2))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 4, 8))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 8, 4))
        layers.append(FourDilateConvResBlockIN(in_channels, out_channels, 2, 1))
        return nn.Sequential(*layers)


class Ends(nn.Module):
    def __init__(self):
        super(Ends, self).__init__()
        self.generateFrame_first = N8_IN()
        self.generateFrame_last = N8_IN()
        self.regenerateFrame_first = N8_IN()

    def forward(self, Blurry, ref4):
        ref1 = self.generateFrame_first(Blurry, ref4, ref4) + ref4
        ref7 = self.generateFrame_last(Blurry, ref4, ref1) + ref4
        ref1 = self.regenerateFrame_first(Blurry, ref4, ref7) + ref4

        return ref1, ref7