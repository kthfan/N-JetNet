import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_same_padding_size
from .gaussian_basis_filters import GaussianBasisFiltersShared


__all__ = [
    "SrfConv2d",
    "SrfConvTranspose2d",
    "safe_subsample",
    "safe_upsample",
]

class SrfConv2d(nn.Module):
    """ The N-Jet convolutional-layer using a linear combination of
    Gaussian derivative filters.
    Inputs:
        - in_channels: input channels
        - out_channels: output channels
        - stride: the stride of the convolving kernel. Can be a single number or a tuple (sH, sW). (default: 1)
        - padding: Padding added to all four sides of the input. (default: "SAME")
        - init_k: the spatial extent of the kernels (default: 2)
        - init_order: the order of the approximation (default: 3)
        - init_scale: the initial starting scale, where: sigma=2^scale (default: 0)
        - max_filter_size: the maximum filter size will be 2*max_filter_size + 1 (default: 15).
        - learn_sigma: whether sigma is learnable
        - bias:  If True, adds a learnable bias to the output. (default: True)
        - groups: groups for the convolution (default: 1)
        - ssample: if we subsample the featuremaps based on sigma (default: False)
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 padding="SAME",
                 init_k=2,
                 init_order=3,
                 init_scale=0,
                 max_filter_size=15,
                 learn_sigma=True,
                 bias=True,
                 groups=1,
                 ssample=False,
                 device=None,
                 dtype=None):
        super(SrfConv2d, self).__init__()

        self.init_k = init_k
        self.init_order = init_order
        self.init_scale = init_scale
        self.max_filter_size = max_filter_size
        self.in_channels = in_channels
        self.ssample = ssample

        assert (out_channels % groups == 0)
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride
        self.padding = padding.upper() if isinstance(padding, str) else padding

        """ Define the number of basis based on order. """
        F = int((self.init_order + 1) * (self.init_order + 2) / 2)
        
        self.guass_basis_filters_shared = GaussianBasisFiltersShared(max_filter_size=self.max_filter_size,
                                                                     max_order=self.init_order+1,
                                                                     kernel_scale=self.init_k)

        """ Create weight variables. """
        self.weight = torch.nn.Parameter(torch.zeros([F, int(in_channels / groups), out_channels],
                                                     dtype=dtype, device=device), requires_grad=True)
        torch.nn.init.normal_(self.weight, mean=0.0, std=1)

        """ Define the scale parameter. """
        self.scales = torch.nn.Parameter(self.init_scale * torch.ones([1], dtype=dtype, device=device),
                                         requires_grad=learn_sigma)

        """ Define the bias parameter. """
        self.bias = torch.nn.Parameter(torch.zeros([out_channels], dtype=dtype, device=device),
                                       requires_grad=True) if bias else None

    def forward(self, data):
        """ Forward pass with inputs: creates the filters and performs the convolution. """

        """ Define sigma from the scale: sigma = 2^scale """
        sigmas = 2.0 ** self.scales
        xy_basis = self.guass_basis_filters_shared(sigmas)
        filters = torch.einsum('fio, nfhw -> noihw', self.weight, xy_basis)

        """ Subsample based on sigma if wanted. """
        if self.ssample:
            data = safe_subsample(data, sigmas.mean())

        """ Compute padding size """
        if isinstance(self.padding, str):
            if self.padding == "SAME":
                padding_sizes = (get_same_padding_size(filters.shape[-2], self.stride),
                                get_same_padding_size(filters.shape[-1], self.stride))
            elif self.padding == "VALID":
                padding_sizes = (0, 0)
        else:
            padding_sizes = (self.padding, self.padding) if isinstance(self.padding, int) else self.padding

        strides = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride
        """ Perform the convolution. """
        final_conv = F.conv2d(
            input=data,  # NCHW
            weight=filters[0],  # KCHW
            bias=self.bias,
            stride=strides,
            padding=padding_sizes,
            groups=self.groups)

        return final_conv


class SrfConvTranspose2d(nn.Module):
    """ The N-Jet convolutional-layer for transpose conv2d.
    Inputs:
        - in_channels: input channels
        - out_channels: output channels
        - stride: the stride of the convolving kernel. Can be a single number or a tuple (sH, sW). (default: 1)
        - padding: Padding added to all four sides of the input. (default: "SAME")
        - output_padding: Additional size added to one side of each dimension in the output shape. (default: 0)
        - init_k: the spatial extent of the kernels (default: 2)
        - init_order: the order of the approximation (default: 3)
        - init_scale: the initial starting scale, where: sigma=2^scale (default: 0)
        - max_filter_size: the maximum filter size will be 2*max_filter_size + 1 (default: 15).
        - learn_sigma: whether sigma is learnable
        - bias:  If True, adds a learnable bias to the output. (default: True)
        - groups: groups for the convolution (default: 1)
        - ssample: if we upsample the featuremaps based on sigma (default: False)
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 padding="SAME",
                 output_padding=0,
                 init_k=2,
                 init_order=3,
                 init_scale=0,
                 max_filter_size=15,
                 learn_sigma=True,
                 bias=True,
                 groups=1,
                 ssample=False,
                 device=None,
                 dtype=None):
        super(SrfConvTranspose2d, self).__init__()

        self.init_k = init_k
        self.init_order = init_order
        self.init_scale = init_scale
        self.max_filter_size = max_filter_size
        self.in_channels = in_channels
        self.ssample = ssample

        assert (out_channels % groups == 0)
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride
        self.padding = padding.upper() if isinstance(padding, str) else padding
        self.output_padding = output_padding

        """ Define the number of basis based on order. """
        F = int((self.init_order + 1) * (self.init_order + 2) / 2)

        self.guass_basis_filters_shared = GaussianBasisFiltersShared(max_filter_size=self.max_filter_size,
                                                                     max_order=self.init_order + 1,
                                                                     kernel_scale=self.init_k)

        """ Create weight variables. """
        self.weight = torch.nn.Parameter(torch.zeros([F, int(out_channels / groups), in_channels],
                                                     dtype=dtype, device=device), requires_grad=True)
        torch.nn.init.normal_(self.weight, mean=0.0, std=1)

        """ Define the scale parameter. """
        self.scales = torch.nn.Parameter(self.init_scale * torch.ones([1], dtype=dtype, device=device),
                                         requires_grad=learn_sigma)

        """ Define the bias parameter. """
        self.bias = torch.nn.Parameter(torch.zeros([out_channels], dtype=dtype, device=device),
                                       requires_grad=True) if bias else None


    def forward(self, data):
        """ Forward pass with inputs: creates the filters and performs the convolution. """

        """ Define sigma from the scale: sigma = 2^scale """
        sigmas = 2.0 ** self.scales
        xy_basis = self.guass_basis_filters_shared(sigmas)
        filters = torch.einsum('foi, nfhw -> niohw', self.weight, xy_basis)

        """ Subsample based on sigma if wanted. """
        if self.ssample:
            data = safe_upsample(data, sigmas.mean())

        """ Compute padding size """
        if isinstance(self.padding, str):
            if self.padding == "SAME":
                padding_sizes = (get_same_padding_size(filters.shape[-2], self.stride),
                                 get_same_padding_size(filters.shape[-1], self.stride))
            elif self.padding == "VALID":
                padding_sizes = (0, 0)
        else:
            padding_sizes = (self.padding, self.padding) if isinstance(self.padding, int) else self.padding

        strides = (self.stride, self.stride) if isinstance(self.stride, int) else self.stride

        """ Perform the convolution. """
        final_conv = F.conv_transpose2d(
            input=data,  # NCHW
            weight=filters[0],  # KCHW
            bias=self.bias,
            stride=strides,
            padding=padding_sizes,
            output_padding=self.output_padding,
            groups=self.groups)

        return final_conv


def safe_subsample(current, sigma, r=4.0):
    """ Subsampling of the featuremaps based on the learned sigma.
    Input:
        - current: input featuremap
        - sigma: the learned sigma values
        - r: the hyperparameter controlling how fast the subsampling goes as a function of sigma.
    """
    scale = min(1.0, torch.div(r, 2**sigma))
    shape = current.shape
    shape_out = max([1, 1], [int(float(shape[2]) * scale), \
                             int(float(shape[3]) * scale)])
    current_out = F.interpolate(current, shape_out)
    return current_out

def safe_upsample(current, sigma, r=4.0):
    """ Upsampling of the featuremaps based on the learned sigma.
    Input:
        - current: input featuremap
        - sigma: the learned sigma values
        - r: the hyperparameter controlling how fast the upsampling goes as a function of sigma.
    """
    scale = max(1.0, r / 2**sigma)
    shape = current.shape
    shape_out = max([1, 1], [int(float(shape[2]) * scale), \
                             int(float(shape[3]) * scale)])
    current_out = F.interpolate(current, shape_out)
    return current_out



