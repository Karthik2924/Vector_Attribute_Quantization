import torch
import torch.nn as nn
import numpy as np

from . import base

class DenseDecoder( nn.Module,base.Decoder):
    def __init__(self, dense_partial, out_shape, x):
        super(DenseDecoder, self).__init__()

        # Assuming dense_partial is a PyTorch module
        self.dense = dense_partial(x)

        # Assuming dense_partial returns a tensor
        x = self.dense(x)

        # Assuming eqx.nn.Linear is equivalent to torch.nn.Linear
        self.head = nn.Linear(x.shape[0], np.prod(out_shape))

        self.to_output = nn.Sequential(
            nn.Flatten(),
            nn.Lambda(lambda x: x.view(*out_shape))
        )

    def forward(self, z):
        x = self.dense(z)
        x = self.head(x)
        x_hat_logits = self.to_output(x)

        outs = {
            'x_hat_logits': x_hat_logits
        }
        return outs
