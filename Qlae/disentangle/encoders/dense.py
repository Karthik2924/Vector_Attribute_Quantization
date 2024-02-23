import torch
import torch.nn as nn

import disentangle
from . import base


class DenseEncoder(base.Encoder):
    def __init__(self, dense_partial, x, out_size):
        super(DenseEncoder, self).__init__()
        self.to_dense = nn.Flatten()
        x = self.to_dense(x)
        self.dense = dense_partial(x)
        x = self.dense(x)
        self.head = nn.Linear(x.shape[0], out_size)

    def forward(self, x):
        x = self.to_dense(x)
        features = self.dense(x)
        pre_z = self.head(features)
        outs = {
            'pre_z': pre_z,
            'features': features
        }
        return outs
