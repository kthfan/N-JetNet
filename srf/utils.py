
__all__ = [
    "get_same_padding_size",
]

def get_same_padding_size(kernel_size, stride):
    """
         nn.Conv2d(in_ch, out_ch, kernel_size, stride, get_same_padding_size(kernel_size, stride))
         is equivalent to nn.Conv2d(in_ch, out_ch, kernel_size, stride, "SAME").
         IMPORTENT: This only works when feature size is even and kernel_size is odd.
    """
    padding = (kernel_size - stride) // 2
    padding = padding + (2 * padding - kernel_size) % stride
    return padding