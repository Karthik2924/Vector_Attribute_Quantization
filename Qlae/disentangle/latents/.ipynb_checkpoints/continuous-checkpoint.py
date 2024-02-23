import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from . import base

class ContinuousLatent(nn.Module, base.Latent):
    def __init__(self, num_latents):
        super(ContinuousLatent, self).__init__()
        self.is_continuous = True
        self.num_latents = num_latents
        self.num_inputs = num_latents

    def forward(self, x):
        outs = {
            'z_hat': x
        }
        return outs

class StandardGaussianLatent(ContinuousLatent):
    def sample(self):
        return torch.randn(self.num_latents)

class UniformLatent(ContinuousLatent):
    def sample(self):
        return torch.rand(self.num_latents)

# def _test():
#     continuous_latent = ContinuousLatent(num_latents=10)
#     standard_gaussian_latent = StandardGaussianLatent(num_latents=10)
#     uniform_latent = UniformLatent(num_latents=10)

#     x = torch.ones((1, 10))  # Replace with actual input size

#     # Forward pass
#     continuous_outs = continuous_latent(x)
#     gaussian_outs = standard_gaussian_latent(x)
#     uniform_outs = uniform_latent(x)

#     # Sample from distributions
#     sample_gaussian = gaussian_outs['z_hat']
#     sample_uniform = uniform_outs['z_hat']

# if __name__ == '__main__':
#     _test()
