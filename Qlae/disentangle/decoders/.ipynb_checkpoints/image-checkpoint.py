import torch
import torch.nn as nn
import numpy as np

from . import base

class ImageDecoder( base.Decoder):
    def __init__(self, dense_partial, transition_partial, conv_transpose_partial, conv_partial, out_shape, x, transition_shape):
        super(ImageDecoder, self).__init__()

        # Assuming dense_partial is a PyTorch module
        dense = dense_partial(x)

        # Assuming dense_partial returns a tensor
        x = dense(x)

        # Assuming transition_partial is a PyTorch module
        transition = transition_partial(x.shape[0], np.prod(transition_shape))
        x = transition(x)

        dense_to_conv = nn.Sequential(
            nn.Flatten(),
            nn.Lambda(lambda x: x.view(*transition_shape))
        )

        conv_transpose = conv_transpose_partial(x)

        # Assuming conv_partial is a PyTorch module
        conv = conv_partial(x.shape[0], out_shape[0])

        self.transition_shape = transition_shape

        self.layers = nn.Sequential(
            dense,
            transition,
            dense_to_conv,
            conv_transpose,
            conv
        )

    def forward(self, z):
        x = self.layers(z)

        outs = {
            'x_hat_logits': x
        }
        return outs
