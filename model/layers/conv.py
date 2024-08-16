import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.layers import conv3x3

def subpel_conv1x1(in_ch, out_ch, r=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r)
    )


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )

