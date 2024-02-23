import torch.nn as nn
import functools

def append_normalization(layers, norm, **kwargs):
    if norm == 'instance_norm':
        layers.append(nn.GroupNorm(num_groups=kwargs['out_channels'], num_channels=kwargs['out_channels'], eps=1e-6))
    elif norm == 'layer_norm':
        layers.append(nn.LayerNorm(kwargs['shape']))
    elif norm == 'none':
        pass
    else:
        raise ValueError(f'unknown norm: {norm}')
    return layers

def append_activation(layers, activation, **kwargs):
    if activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'leaky_relu':
        layers.append(nn.LeakyReLU(negative_slope=0.2))
    elif activation == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'none':
        pass
    else:
        raise ValueError(f'unknown activation: {activation}')
    return layers
