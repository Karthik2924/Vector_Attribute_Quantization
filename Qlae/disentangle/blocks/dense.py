import torch
import torch.nn as nn
from . import utils, base

class DenseBlock(base.Block):
    def __init__(self, in_size, out_size, norm='none', activation='relu', use_bias=True):
        super(DenseBlock, self).__init__()
        layers = []
        layers.append(nn.Linear(in_size, out_size, bias=use_bias))
        layers = utils.append_normalization(layers, norm, shape=(out_size,))
        layers = utils.append_activation(layers, activation)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class DenseBlocks( base.Block):

    def __init__(self, widths, activation, norm,in_size=1):
        super(DenseBlocks, self).__init__()
        blocks = []
        for i, width in enumerate(widths):
            block = DenseBlock(
                in_size=widths[i - 1] if i > 0 else in_size,  # Assuming 1 for the input size in the first block
                out_size=width,
                activation=activation,
                norm=norm
            )
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)
