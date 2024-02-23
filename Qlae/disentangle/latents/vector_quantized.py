import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizedLatent(nn.Module):
    def __init__(self, num_latents, num_embeddings, embedding_size, key):
        super(VectorQuantizedLatent, self).__init__()

        values_key, _ = torch.split(key, 2)
        self.is_continuous = False
        self.num_latents = num_latents
        self.num_inputs = num_latents * embedding_size

        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        s = 1.0 / self.embedding_size
        limit = torch.sqrt(3.0 * s)
        self.embeddings = nn.Parameter(
            torch.rand(num_embeddings, embedding_size) * 2 * limit - limit
        )

    def quantize(self, x):
        squared_distances = torch.sum((x - self.embeddings)**2, dim=1)
        index = torch.argmin(squared_distances)
        return self.embeddings[index], index

    def forward(self, x):
        x_ = x.view(self.num_latents, self.embedding_size)
        quantized, indices = torch.jit.vmap(self.quantize)(x_)
        quantized = quantized.view(-1)
        quantized_sg = x + quantized - x.detach()
        outs = {
            'z_continuous': x,
            'z_quantized': quantized,
            'z_hat': quantized_sg,
            'z_indices': indices
        }
        return outs

    def sample(self):
        ret = []
        for _ in range(self.num_latents):
            indices = torch.randint(0, self.num_embeddings, (self.embedding_size,))
            values = self.embeddings[indices]
            ret.append(values)
        return torch.stack(ret)

# def _test():
#     num_latents = 10
#     num_embeddings = 5
#     embedding_size = 8
#     key = torch.randn(2)

#     vql = VectorQuantizedLatent(num_latents, num_embeddings, embedding_size, key)
#     x = torch.ones((1, num_latents * embedding_size))

#     # Forward pass
#     outputs = vql(x)

#     # Accessing outputs
#     z_continuous = outputs['z_continuous']
#     z_quantized = outputs['z_quantized']
#     z_hat = outputs['z_hat']
#     z_indices = outputs['z_indices']

#     # Sample
#     sample = vql.sample()

# if __name__ == '__main__':
#     _test()
