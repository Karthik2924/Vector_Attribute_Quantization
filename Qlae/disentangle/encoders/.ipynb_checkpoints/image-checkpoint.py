import torch
import torch.nn as nn
import jax
import disentangle
from . import base

class ImageEncoder(base.Encoder):
    conv: nn.Module
    conv_to_dense: nn.Module
    dense: nn.Module
    head: nn.Module

    def __init__(self, conv_partial, dense_partial, *, x, out_size, key):
        super(ImageEncoder, self).__init__()
        
        # Split the key
        #conv_key, dense_key, head_key = torch.chunk(key, 3)
        conv_key, dense_key, head_key = jax.random.split(key, 3)

        # Convolutional layer
        print(conv_partial)
        self.conv = conv_partial(x=x, key=conv_key)

        # Transition shape
        x = self.conv(x)
        self.transition_shape = x.shape

        # Reshape to a vector
        self.conv_to_dense = nn.Flatten()

        # Dense layer
        x = self.conv_to_dense(x)
        self.dense = dense_partial(x=x, key=dense_key)

        # Head linear layer
        x = self.dense(x)
        self.head = nn.Linear(x.shape[0], out_size)

    def forward(self, x, *, key=None):
        x = self.conv(x)
        x = self.conv_to_dense(x)
        features = self.dense(x)
        pre_z = self.head(features)

        outs = {
            'pre_z': pre_z,
            'features': features
        }

        return outs
