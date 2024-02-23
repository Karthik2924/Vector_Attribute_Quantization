import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from . import base

class DiagonalGaussianLatent(nn.Module, base.Latent):
    def __init__(self, num_latents):
        super(DiagonalGaussianLatent, self).__init__()
        self.is_continuous = True
        self.num_latents = num_latents
        self.num_inputs = 2 * num_latents

    def forward(self, x):
        mu, sigma_param = torch.split(x, self.num_latents, dim=1)
        sigma = torch.nn.functional.softplus(sigma_param)
        z_sample = torch.randn_like(mu) * sigma + mu

        outs = {
            'z_hat': mu,
            'z_sample': z_sample,
            'z_sigma': sigma,
            'z_mu': mu
        }
        return outs

# def _test():
#     diagonal_gaussian_latent = DiagonalGaussianLatent(num_latents=10)
#     x = torch.ones((1, 20))  # Replace with actual input size

#     # Forward pass
#     outputs = diagonal_gaussian_latent(x)

#     # Accessing outputs
#     z_hat = outputs['z_hat']
#     z_sample = outputs['z_sample']
#     z_sigma = outputs['z_sigma']
#     z_mu = outputs['z_mu']

# if __name__ == '__main__':
#     _test()
