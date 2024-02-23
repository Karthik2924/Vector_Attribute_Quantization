import torch
import torch.nn as nn
from . import utils, base

class Conv2DBlock( base.Block):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, norm='instance_norm',
                 activation='leaky_relu', use_bias=False):
        super(Conv2DBlock, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=use_bias)
        )
        layers = utils.append_normalization(layers, norm, out_channels=out_channels)
        layers = utils.append_activation(layers, activation)
        self.block = nn.Sequential(*layers)
        #super().__init__(layers)

    def forward(self, x):
        #print("conv")
        return self.block(x)

class Conv2DBlocks( base.Block):
    def __init__(self, widths, kernel_sizes, strides, paddings, activation, norm,inchannels = 3):
        super(Conv2DBlocks, self).__init__()
        blocks = []
        assert len(widths) == len(kernel_sizes) == len(strides) == len(paddings)
        for i, (width, kernel_size, stride, padding) in enumerate(zip(widths, kernel_sizes, strides, paddings)):
            block = Conv2DBlock(
                in_channels=widths[i - 1] if i > 0 else inchannels,  # Assuming 1 for the input channel in the first block
                out_channels=width,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation=activation,
                norm=norm
            )
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)
