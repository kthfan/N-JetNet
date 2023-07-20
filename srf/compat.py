import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_same_padding_size
from .structured_conv_layer import SrfConv2d, SrfConvTranspose2d

__all__ = [
    "SrfConv2dCompat",
    "SrfConvTranspose2dCompat",
    "hijack_torch_conv2d",
    "restore_hijack_torch_conv2d",
]

class SrfConv2dCompat(nn.Module):
    """ Compatible with torch.nn.Conv2d """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros',
                 device=None, dtype=None, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        if kernel_size == 1:
            self.conv = Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                               dilation=dilation, groups=groups, bias=bias,
                               padding_mode=padding_mode)  # device=device, dtype=dtype)
        else:
            same_padding_size = get_same_padding_size(kernel_size, stride)
            if same_padding_size == padding:
                padding = 'SAME'
            self.conv = SrfConv2d(in_channels, out_channels, stride=stride, padding=padding,
                                  groups=groups, bias=bias, device=device, dtype=dtype, **kwargs)
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        self.transposed = False

    def forward(self, *args, **kwargs):
        return self.conv(*args, **kwargs)


class SrfConvTranspose2dCompat(nn.Module):
    """ Compatible with torch.nn.ConvTranspose2d """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 device=None, dtype=None, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        if kernel_size == 1:
            self.conv = ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        output_padding=output_padding, dilation=dilation, groups=groups, bias=bias,
                                        padding_mode=padding_mode)  # device=device, dtype=dtype)
        else:
            same_padding_size = get_same_padding_size(kernel_size, stride)
            if same_padding_size == padding:
                padding = 'SAME'
            self.conv = SrfConvTranspose2d(in_channels, out_channels, stride=stride, padding=padding,
                                           output_padding=output_padding,
                                           groups=groups, bias=bias, device=device, dtype=dtype, **kwargs)
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        self.transposed = True

    def forward(self, *args, **kwargs):
        return self.conv(*args, **kwargs)


""" backup original Conv2d and ConvTranspose2d """
Conv2d = nn.Conv2d
ConvTranspose2d = nn.ConvTranspose2d

def hijack_torch_conv2d():
    """ Replace Conv2d and ConvTranspose2d with SrfConv2d and SrfConvTranspose2d"""
    nn.Conv2d = SrfConv2dCompat
    nn.ConvTranspose2d = SrfConvTranspose2dCompat


def restore_hijack_torch_conv2d():
    """ Undo the replacement of SrfConv2d and SrfConvTranspose2d"""
    nn.Conv2d = Conv2d
    nn.ConvTranspose2d = ConvTranspose2d
