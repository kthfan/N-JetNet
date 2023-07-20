import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "GaussianBasisFiltersShared"
]

class GaussianBasisFiltersShared(nn.Module):
    """ Compute Gaussian derivative basis, which shape is [batch_size, (order + 1) * (order + 2) / 2, kernel height, kernel width].
        - max_filter_size: the maximum filter size will be 2*max_filter_size + 1 (default: 15).
        - max_order: the max order of Gaussian derivative for the approximation (default: 4)
        - kernel_scale: the spatial extent of the kernels (default: 2)
    """
    def __init__(self, max_filter_size=15, max_order=4, kernel_scale=2):
        super().__init__()
        self.max_filter_size = max_filter_size
        self.max_order = max_order
        self.kernel_scale = kernel_scale

        # Define variables:
        #  x: the grid
        #  orders: order of Hermite polynomials
        #  coefs: order of Hermite polynomials, a max_order * max_order matrix
        x = torch.arange(-max_filter_size, max_filter_size + 1, 1, dtype=torch.float32)
        orders = torch.arange(0, max_order, 1, dtype=torch.float32)
        coefs = self._get_hermite_coefs(max_order)

        # Reshape to [batch_size, max_order, terms of Hermite polynomials, filter size]
        x = x.view(1, 1, 1, 2 * max_filter_size + 1)
        coefs = coefs.view(1, max_order, max_order, 1)

        # All terms of Hermite polynomials of x
        x_hermites = coefs * (x / 2 ** 0.5) ** orders.view(1, 1, -1, 1)

        # Indices which are used to build 2d Gaussian derivative filters.
        x_indices, y_indices = list(zip(*[(i, j) for i in range(max_order) for j in range(max_order-i-1, -1, -1)]))
        self.x_indices, self.y_indices = torch.LongTensor(x_indices), torch.LongTensor(y_indices)

        self.register_buffer('x', x, persistent=False)
        self.register_buffer('x_hermites', x_hermites, persistent=False)
        self.register_buffer('orders', orders, persistent=False)

    def forward(self, sigmas):
        # Compute filter size for each sigmas and select the biggest filter size
        filter_sizes = torch.ceil(self.kernel_scale * sigmas + 0.5).detach().cpu().int()
        filter_size = torch.max(filter_sizes)
        filter_size = min(filter_size, self.max_filter_size)

        # Reshape to [batch_size, max_order, terms of Hermite polynomials, filter size]
        sigmas = sigmas.view(-1, 1, 1, 1)
        # Clip size of x and x_hermites according to filter_size.
        x = self.x[..., self.max_filter_size - filter_size: self.max_filter_size + filter_size + 1]
        x_hermites = self.x_hermites[..., self.max_filter_size - filter_size: self.max_filter_size + filter_size + 1]

        # The 0th order Gaussian derivatives.
        gauss = 1 / ((2 * math.pi) ** 0.5 * sigmas) * torch.exp(- x ** 2 / (2 * sigmas ** 2))
        gauss = gauss / gauss.sum(dim=3, keepdim=True)

        # All terms of Hermite polynomials of sigmas.
        sigma_hermites = sigmas ** self.orders.view(1, 1, -1, 1)
        # All terms of Hermite polynomials of $x / 2 \sigma$.
        hermites = torch.sum(x_hermites / sigma_hermites, dim=2, keepdim=True) # [batch_size, max_order, 1, filter size]

        # Compute 0-(max_order-1)th order Gaussian derivatives
        basis = (-1.0 / (2 ** 0.5 * sigmas)) ** self.orders.view(1, -1, 1, 1) * hermites * gauss
        basis = basis * sigmas ** self.orders.view(1, -1, 1, 1) # normalize
        basis = basis.squeeze(dim=2) # [batch_size, max_order, filter size]

        # Compute 2d Gaussian derivatives.
        x_basis, y_basis = basis[:, :, None, :], basis[:, :, :, None]
        x_basis, y_basis = x_basis[:, self.x_indices], y_basis[:, self.y_indices]
        xy_basis = x_basis * y_basis  # [batch_size, filters, kH, kW]

        return xy_basis

    def _get_hermite_coefs(self, order):
        """ Returns a $order \times order$ matrix, which i-st row represent coefficients of i-st order
            hermite polynomial arranged in ascending orde.
        """
        coefs = torch.zeros(order, order, dtype=torch.float32)
        coefs[0, 0] = 1
        for t in range(1, order):
            coefs[t, 0] = - coefs[t - 1, 1]
            for s in range(1, t + 1):
                coefs[t, s] = 2 * t / s * coefs[t - 1, s - 1]
        return coefs




