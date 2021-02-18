import torch
import torch.nn as nn
from numpy.random import normal
from numpy.linalg import svd
from math import sqrt


def _get_orthogonal_init_weights(weights):
    fan_out = weights.size(0)
    fan_in = weights.size(1) * weights.size(2) * weights.size(3)

    u, _, v = svd(normal(0.0, 1.0, (fan_out, fan_in)), full_matrices=False)

    if u.shape == (fan_out, fan_in):
        return torch.Tensor(u.reshape(weights.size()))
    else:
        return torch.Tensor(v.reshape(weights.size()))


def pixel_reshuffle(input, upscale_factor):
    """Rearranges elements in a tensor of shape ``[*, C, H, W]`` to a
    tensor of shape ``[C*r^2, H/r, W/r]``.

    See :class:`~torch.nn.PixelShuffle` for details.

    Args:
        input (Variable): Input
        upscale_factor (int): factor to increase spatial resolution by

    Examples:
        >>> input = autograd.Variable(torch.Tensor(1, 3, 12, 12))
        >>> output = pixel_reshuffle(input,2)
        >>> print(output.size())
        torch.Size([1, 12, 6, 6])
    """
    batch_size, channels, in_height, in_width = input.size()

    # // division is to keep data type unchanged. In this way, the out_height is still int type
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor
    input_view = input.contiguous().view(batch_size, channels, out_height, upscale_factor, out_width, upscale_factor)
    channels = channels * upscale_factor * upscale_factor

    shuffle_out = input_view.permute(0,1,3,5,2,4).contiguous()
    return shuffle_out.view(batch_size, channels, out_height, out_width)


class FourDilateConvResBlockIN(nn.Module):
    def __init__(self, in_channels, out_channels, dilation2, dilation4):
        super(FourDilateConvResBlockIN, self).__init__()
        self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.norm2 = nn.InstanceNorm2d(in_channels, affine=True)
        self.norm4 = nn.InstanceNorm2d(in_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels,  out_channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.conv2 = nn.Conv2d(in_channels,  out_channels, (3, 3), (1, 1), (dilation2, dilation2), (dilation2, dilation2), bias=False)
        self.conv4 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (dilation4, dilation4), (dilation4, dilation4), bias=False)

    def forward(self, x):
        out1 = self.norm1(x)
        out1 = self.relu(out1)
        out1 = self.conv1(out1)

        out2 = self.norm2(x)
        out2 = self.relu(out2)
        out2 = self.conv2(out2)
        out2 = self.norm4(out2)
        out2 = self.relu(out2)
        out2 = self.conv4(out2)
        out  = x + out1 + out2
        return out

    def _initialize_weights(self):
        self.conv1.weight.data.copy_(_get_orthogonal_init_weights(self.conv1.weight))
        self.conv2.weight.data.copy_(_get_orthogonal_init_weights(self.conv2.weight))
        self.conv4.weight.data.copy_(_get_orthogonal_init_weights(self.conv4.weight))


class FourConvResBlockIN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FourConvResBlockIN, self).__init__()
        self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.norm2 = nn.InstanceNorm2d(in_channels, affine=True)
        self.norm4 = nn.InstanceNorm2d(in_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)

    def forward(self, x):
        out1 = self.norm1(x)
        out1 = self.relu(out1)
        out1 = self.conv1(out1)

        out2 = self.norm2(x)
        out2 = self.relu(out2)
        out2 = self.conv2(out2)
        out2 = self.norm4(out2)
        out2 = self.relu(out2)
        out2 = self.conv4(out2)
        out  = x + out1 + out2
        return out

    def _initialize_weights(self):
        self.conv1.weight.data.copy_(_get_orthogonal_init_weights(self.conv1.weight))
        self.conv2.weight.data.copy_(_get_orthogonal_init_weights(self.conv2.weight))
        self.conv4.weight.data.copy_(_get_orthogonal_init_weights(self.conv4.weight))


class FourConvResBlockINDis(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FourConvResBlockINDis, self).__init__()
        self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.norm2 = nn.InstanceNorm2d(in_channels, affine=True)
        self.norm4 = nn.InstanceNorm2d(in_channels, affine=True)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)

    def forward(self, x):
        out1 = self.norm1(x)
        out1 = self.lrelu(out1)
        out1 = self.conv1(out1)

        out2 = self.norm2(x)
        out2 = self.lrelu(out2)
        out2 = self.conv2(out2)
        out2 = self.norm4(out2)
        out2 = self.lrelu(out2)
        out2 = self.conv4(out2)
        out  = x + out1 + out2
        return out

    def _initialize_weights(self):
        self.conv1.weight.data.copy_(_get_orthogonal_init_weights(self.conv1.weight))
        self.conv2.weight.data.copy_(_get_orthogonal_init_weights(self.conv2.weight))
        self.conv4.weight.data.copy_(_get_orthogonal_init_weights(self.conv4.weight))


class TwoConvBlockIN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TwoConvBlockIN, self).__init__()
        self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.norm2 = nn.InstanceNorm2d(in_channels, affine=True)

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)

    def forward(self, x):
        out1 = self.norm1(x)
        out1 = self.relu(out1)
        out1 = self.conv1(out1)


        out2 = self.norm2(x)
        out2 = self.relu(out2)
        out2 = self.conv2(out2)

        out  = out1 + out2
        return out

    def _initialize_weights(self):
        self.conv1.weight.data.copy_(_get_orthogonal_init_weights(self.conv1.weight))
        self.conv2.weight.data.copy_(_get_orthogonal_init_weights(self.conv2.weight))