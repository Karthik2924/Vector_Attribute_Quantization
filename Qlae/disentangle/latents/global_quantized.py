import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from . import base

class GlobalQuantizedLatent(nn.Module, base.Latent):
    def __init__(self, num_latents, num_values):
        super(GlobalQuantizedLatent, self).__init__()
        self.is_continuous = False
        self.num_latents = num_latents
        self.num_inputs = num_latents
        self.num_values = num_values
        self.values = torch.linspace(-0.5, 0.5, num_values)

    @staticmethod
    def quantize(x, values):
        distances = torch.abs(x - values)
        _, index = distances.min(dim=0)
        return values[index], index

    def forward(self, x):
        quantized, indices = torch.jit.annotate(List[Tensor], [])
        for xi in x:
            q, i = self.quantize(xi, self.values)
            quantized.append(q)
            indices.append(i)

        quantized = torch.stack(quantized)
        indices = torch.stack(indices)

        quantized_sg = x + torch.autograd.Variable(quantized - x, requires_grad=False)
        outs = {
            'z_continuous': x,
            'z_quantized': quantized,
            'z_hat': quantized_sg,
            'z_indices': indices
        }

        return outs

# def _test():
#     global_quantized_latent = GlobalQuantizedLatent(num_latents=10, num_values=100)
#     x = torch.ones((1, 10))  # Replace with actual input size

#     # Forward pass
#     outputs = global_quantized_latent(x)

#     # Accessing outputs
#     z_continuous = outputs['z_continuous']
#     z_quantized = outputs['z_quantized']
#     z_hat = outputs['z_hat']
#     z_indices = outputs['z_indices']

# if __name__ == '__main__':
#     _test()
