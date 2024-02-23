import torch
import torch.nn as nn

import disentangle
from . import base


class CriticEncoder(base.Encoder):
    def __init__(self, encoder_base):
        super(CriticEncoder, self).__init__()
        self.base = encoder_base

        # Assuming encoder_base returns a dictionary with 'features'
        out_features = self.base(torch.randn(1, 1))['features']
        self.critic = nn.Linear(out_features.shape[1], 1)

    def forward(self, x):
        outs = self.base(x)
        value = self.critic(outs['features'])
        outs['value'] = value
        return outs

    def value(self, x):
        outs = self.forward(x)
        return torch.squeeze(outs['value'])

    @property
    def transition_shape(self):
        return self.base.transition_shape
