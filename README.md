# N-JetNet
PyTorch implementation of "Resolution learning in deep convolutional networks using scale-space theory", Silvia L.Pintea, Nergis Tomen, Stanley F. Goes, Marco Loog, Jan C. van Gemert, Transactions on Image Processing, 2021.
This implementation is approximately 1.18 times faster than the [official code](https://github.com/SilviaLauraPintea/N-JetNet/").

## Benchmark
This benchmark is evaluated on resnet18 with stride=1 and without maxpooling in the stem. Batch size is 64, image size is $64\times 64 \times 3$ and optimizer is Adam. 
|  method   | speed (iteration/s)  | memory usage (GB) |
|  ----  | ----  | ---- |
| [Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) | 16 | 2.7 |
| [original](https://github.com/SilviaLauraPintea/N-JetNet) | 4.3 | 2.9 |
| [this](https://github.com/kthfan/N-JetNet/) | 5.1 | 2.9 |

## Usage

##### Use `srf.SrfConv2d`.
```python
import torch
import srf

# Subsampling by stride=2
x = torch.randn(1, 64, 112, 112)
conv = srf.SrfConv2d(64, 128, stride=2, padding='SAME')
print(conv(x).shape)

# Using safe subsample
conv = srf.SrfConv2d(64, 128, stride=1, padding='VALID', ssample=True)
conv.scales.data[0] = 1.5
print(conv(x).shape)
```

##### Use `srf.SrfConvTranspose2d`.
```python
import torch
import srf

# Upsampling by stride=2
x = torch.randn(1, 128, 112, 112)
conv = srf.SrfConvTranspose2d(128, 64, stride=2, padding='SAME', output_padding=1)
print(conv(x).shape)

# Using safe upsample
conv = srf.SrfConvTranspose2d(128, 64, stride=1, padding='VALID', ssample=True)
conv.scales.data[0] = 0
print(conv(x).shape)
```

##### Replace `nn.Conv2d` with `srf.SrfConv2d`.
```python
import torchvision
import srf

srf.hijack_torch_conv2d() # replace nn.Conv2d to srf.SrfConv2d
resnet_w_srf = torchvision.models.resnet50(pretrained=False)

srf.restore_hijack_torch_conv2d() # restore the replacement
resnet_wo_srf = torchvision.models.resnet50(pretrained=False)
```

```python
from torch import nn
import srf

srf.hijack_torch_conv2d() # replace nn.Conv2d to srf.SrfConv2d
print(nn.Conv2d, nn.ConvTranspose2d)


srf.restore_hijack_torch_conv2d() # restore the replacement
print(nn.Conv2d, nn.ConvTranspose2d)
```

##### `srf.GaussianBasisFiltersShared`.
```python
import matplotlib.pyplot as plt
import torch
import srf

sigmas = torch.FloatTensor([1., 2., 3., 5., 6.])
gauss_basis_filters_shared = srf.GaussianBasisFiltersShared(
    max_filter_size = 10,
    max_order = 4,
    kernel_scale = 2,
)
basis = gauss_basis_filters_shared(sigmas)

print('Shape:', basis.shape) # [batch_size, orders, *kernel_size]

i = 3
plt.imshow(basis[2, i])
```

##### Classification task example
```bash
python main.py --dataset <dataset path> --val-dataset <test dataset path> --save-path <model path>
```

