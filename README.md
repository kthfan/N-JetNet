# N-JetNet
Implementation of "Resolution learning in deep convolutional networks using scale-space theory"
The training time of this implementation is approximately 0.74 times of that of "https://github.com/SilviaLauraPintea/N-JetNet/" (tested in resnet 50).

## Usage

##### Use `srf.Conv2d`.
```python
# Subsampling by stride=2
x = torch.randn(1, 64, 112, 112)
conv = srf.SrfConv2d(64, 128, stride=2, padding='SAME')
print(conv(x).shape)

# Using safe subsample
conv = srf.SrfConv2d(64, 128, stride=1, padding='VALID', ssample=True)
conv.scales.data[0] = 1.5
print(conv(x).shape)
```

##### Replace `nn.Conv2d` to `srf.Conv2d`.
```python
import srf

srf.hijack_torch_conv2d() # replace nn.Conv2d to srf.SrfConv2d
resnet_w_srf = torchvision.models.resnet50(pretrained=False)

srf.restore_hijack_torch_conv2d() # restore the replacement
resnet_wo_srf = torchvision.models.resnet50(pretrained=False)
```


