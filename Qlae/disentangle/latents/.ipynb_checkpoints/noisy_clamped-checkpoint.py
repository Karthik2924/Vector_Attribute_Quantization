import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from . import continuous

class NoisyClampedLatent(continuous.ContinuousLatent):
    def __init__(self, clamp, noise, **kwargs):
        super(NoisyClampedLatent, self).__init__(**kwargs)
        self.clamp = clamp
        self.noise = noise

    def forward(self, x):
        x = torch.clamp(x, -self.clamp, self.clamp)
        x = x + torch.randn_like(x) * self.noise
        outs = {
            'z_hat': x
        }
        return outs

# def _test():
#     noisy_clamped_latent = NoisyClampedLatent(clamp=0.5, noise=0.1)
#     x = torch.ones((1, 10))  # Replace with actual input size

#     # Forward pass
#     outputs = noisy_clamped_latent(x)

#     # Accessing outputs
#     z_hat = outputs['z_hat']

# if __name__ == '__main__':
#     _test()
