import torch
import torch.nn as nn
import typing
import numpy as np
from . import base

class StyleConvTranspose2DBlock( base.Block):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='leaky_relu', w_size=12):
        super(StyleConvTranspose2DBlock, self).__init__()
        
        self.forward_block = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conditioning = nn.Linear(w_size, 2 * out_channels)
        self.instance_norm = nn.GroupNorm(out_channels, out_channels, eps=1e-6)

        self.activation = self._get_activation(activation)

    def _get_activation(self, activation):
        if activation == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'relu':
            return nn.ReLU()
        else:
            raise ValueError(f'unknown activation: {activation}')

    def forward(self, x, w):
        scale, bias = torch.chunk(self.conditioning(w), 2, dim=1)
        x = self.forward_block(x)
        x = self.instance_norm(x)


        x = x * scale[:, None, None] + bias[:, None, None]
        
        x = self.activation(x)
        return x

class StyleConvTranspose2DBlocks( base.Block):
    def __init__(self, widths, kernel_sizes, strides, paddings, activation, w_size):
        super(StyleConvTranspose2DBlocks, self).__init__()
        self.blocks = nn.ModuleList([
            StyleConvTranspose2DBlock(
                in_channels=widths[i-1] if i > 0 else 1,  # Assuming 1 for the input channel in the first block
                out_channels=width,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation=activation,
                w_size=w_size
            ) for i, (width, kernel_size, stride, padding) in enumerate(zip(widths, kernel_sizes, strides, paddings))
        ])

    def forward(self, x, w):
        for block in self.blocks:
            x = block(x, w)
        return x

def _test():
    block = StyleConvTranspose2DBlock(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=1,
        activation='leaky_relu',
        w_size=12
    )

    x = torch.ones((1, 8, 4, 4))
    w = torch.ones((1, 12))

    block(x, w)

if __name__ == '__main__':
    _test()
