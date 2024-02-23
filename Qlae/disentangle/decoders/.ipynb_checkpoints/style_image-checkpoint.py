import torch
import torch.nn as nn
import numpy as np
import jax
import jax.numpy as jnp
from . import base

class StyleImageDecoder( base.Decoder):
    def __init__(self, dense_partial, style_conv_transpose_partial, conv_partial, input_channels, transition_shape, out_shape):
        super(StyleImageDecoder, self).__init__()

        input_shape = (input_channels,) + transition_shape[1:]
        self.input_map = 0.1 * torch.ones(input_shape, dtype=torch.float32)

        self.dense = dense_partial(x=None, key=None)  # Replace with actual initialization
        w = self.dense(None)

        self.style_conv_transpose = style_conv_transpose_partial(x=self.input_map, w_size=w.shape[0], key=None)  # Replace with actual initialization
        x = self.style_conv_transpose(x=self.input_map, w=w)

        self.conv = conv_partial(in_channels=x.shape[0], out_channels=out_shape[0], key=None)  # Replace with actual initialization
        x = self.conv(x)

        assert x.shape == out_shape

    def forward(self, z):
        w = self.dense(z)
        x = self.style_conv_transpose(x=self.input_map, w=w)
        outs = {
            'x_hat_logits': self.conv(x)
        }
        return outs

# def _test():
#     decoder = StyleImageDecoder(
#         dense_partial=None,  # Replace with actual initialization
#         style_conv_transpose_partial=None,  # Replace with actual initialization
#         conv_partial=None,  # Replace with actual initialization
#         input_channels=3,
#         transition_shape=(64, 64, 3),
#         out_shape=(64, 64, 3)
#     )

#     z = torch.ones((1, 100))  # Replace with actual input size
#     decoder(z)

# if __name__ == '__main__':
#     _test()
