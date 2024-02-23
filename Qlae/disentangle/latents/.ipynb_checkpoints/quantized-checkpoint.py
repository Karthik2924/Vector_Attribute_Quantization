import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from . import base

class QuantizedLatent(base.Latent,nn.Module):
    def __init__(self, num_latents, num_values_per_latent, optimize_values , key = None):
        super(QuantizedLatent, self).__init__()
        #values_key, _ = torch.split(key, 2)
        self.is_continuous = False
        self.num_latents = num_latents
        self.num_inputs = num_latents

        if isinstance(num_values_per_latent, int):
            self.num_values_per_latent = [num_values_per_latent] * num_latents
        else:
            self.num_values_per_latent = num_values_per_latent

        self._values_per_latent = [torch.linspace(-0.5, 0.5, self.num_values_per_latent[i]) for i in range(num_latents)]
        self.optimize_values = optimize_values

    @property
    def values_per_latent(self):
        if self.optimize_values:
            return self._values_per_latent
        else:
            return [torch.autograd.Variable(v, requires_grad=False) for v in self._values_per_latent]

    @staticmethod
    def quantize(x, values):
        distances = torch.abs(x - values)
        index = torch.argmin(distances)
        return values[index], index

    def forward(self, x):
        quantized_and_indices = [self.quantize(x_i, values_i) for x_i, values_i in zip(x, self.values_per_latent)]
        quantized = torch.stack([qi[0] for qi in quantized_and_indices])
        indices = torch.stack([qi[1] for qi in quantized_and_indices])
        quantized_sg = x + torch.autograd.Variable(quantized - x, requires_grad=False)
        outs = {
            'z_continuous': x,
            'z_quantized': quantized,
            'z_hat': quantized_sg,
            'z_indices': indices
        }

        return outs

    def sample(self):
        ret = []
        for values in self.values_per_latent:
            ret.append(torch.choice(values))
        return torch.tensor(ret)

# def _test():
#     quantized_latent = QuantizedLatent(num_latents=10, num_values_per_latent=5, optimize_values=True, key=torch.randn(2))
#     x = torch.ones((1, 10))  # Replace with actual input size

#     # Forward pass
#     outputs = quantized_latent(x)

#     # Accessing outputs
#     z_continuous = outputs['z_continuous']
#     z_quantized = outputs['z_quantized']
#     z_hat = outputs['z_hat']
#     z_indices = outputs['z_indices']

#     # Sample
#     sample = quantized_latent.sample()

# if __name__ == '__main__':
#     _test()
