import functools
import torch
import torch.nn as nn
import jax
import disentangle
import pandas as pd
import yaml
from disentangle.blocks import*
import os
import torch.nn.functional as F
import seaborn as sns
import inspect
device = torch.device('cuda:3')
lsize = 12
def get_expected_parameters(class_instance):
    # Use inspect.signature to get the signature of the class constructor
    signature = inspect.signature(class_instance.__init__)

    # Extract the parameter names from the signature, excluding 'self'
    parameter_names = [param for param in signature.parameters if param != 'self']

    return parameter_names

def initialize_model(class_instance, param_dict):
    # Get the expected parameters of the class
    expected_params = get_expected_parameters(class_instance)

    # Filter out unnecessary parameters from the given dictionary
    valid_params = {param: param_dict[param] for param in expected_params if param in param_dict}

    # Instantiate the class with the valid parameters
    model = class_instance(**valid_params)

    return model


class Encoder(nn.Module):

    def __init__(self, conv_block, dense_block,din = 4096,dout = 256,hout = lsize,nc = 3):
        super(Encoder, self).__init__()
        
        # Split the key
        #conv_key, dense_key, head_key = torch.chunk(key, 3)
        #conv_key, dense_key, head_key = jax.random.split(key, 3)
        # Convolutional layer
        
        self.conv = initialize_model(Conv2DBlocks,conv_block)
        self.conv_to_dense = nn.Flatten()
        dense_block['in_size'] = 4096
        dense_block['out_size'] = 256
        self.dense = initialize_model(DenseBlocks,dense_block)
        self.head = nn.Linear(256,lsize)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_to_dense(x)
        features = self.dense(x)
        pre_z = self.head(features)
        outs = {
            'pre_z': pre_z,
            'features': features
        }
        return outs


# class QuantizedLatent(disentangle.latents.base.Latent,nn.Module):
#     def __init__(self, num_latents, num_values_per_latent, optimize_values , key = None):
#         super(QuantizedLatent, self).__init__()
#         #values_key, _ = torch.split(key, 2)
#         self.is_continuous = False
#         self.num_latents = lsize
#         self.num_inputs = self.num_latents

#         if isinstance(num_values_per_latent, int):
#             self.num_values_per_latent = [num_values_per_latent] * self.num_latents
#         else:
#             self.num_values_per_latent = num_values_per_latent

#         self._values_per_latent = torch.nn.parameter.Parameter(torch.stack([torch.linspace(-0.5, 0.5, self.num_values_per_latent[i]) for i in range(self.num_latents)]))
#         print(self._values_per_latent.shape)
#         self.optimize_values = optimize_values

#     @property
#     def values_per_latent(self):
#         if self.optimize_values:
#             return self._values_per_latent
#         else:
#             return [torch.autograd.Variable(v, requires_grad=False) for v in self._values_per_latent]

#     @staticmethod
#     def quantize(x, values):
#         distances = torch.abs(x - values)
#         index = torch.argmin(distances)
#         return values[index], index

#     # @staticmethod
#     # def quantize_batch(lat, values):
#     #     # Add a singleton dimension to lat for broadcasting
#     #     lat = lat.unsqueeze(2)
        
#     #     # Calculate distances for the entire batch
#     #     distances = torch.abs(lat - values)
        
#     #     # Find the indices of the minimum distances for each row in the batch
#     #     indices = torch.argmin(distances, dim=2)
        
#     #     # Gather the quantized values based on the indices
#     #     quantized_values = torch.gather(values.unsqueeze(0).expand(lat.size(0), -1, -1), 1, indices.unsqueeze(2))
        
#     #     return quantized_values.squeeze(), indices.squeeze()
#     @staticmethod
#     def quantize_batch(lat, values):
#         # Add a singleton dimension to lat for broadcasting
#         lat = lat.unsqueeze(2)
#         # Calculatew distances for the entire batch
#         distances = torch.abs(lat - values)
#         # Find the indices of the minimum distances for each row in the batch
#         indices = torch.argmin(distances, dim=2)
#         # Gather the quantized values based on the indices
#         quantized_values = torch.gather(values.unsqueeze(0).expand(lat.size(0), -1, -1), 1, indices.unsqueeze(2))
#         qvals = values[torch.arange(0,values.size()[0]),indices]
#         return qvals.squeeze(), indices.squeeze()

    # def forward(self, x):
    #     #print("**forward x requires grad?", x.requires_grad)
    #     #quantized_and_indices = [self.quantize(x_i, values_i) for x_i, values_i in zip(x, self.values_per_latent)]
    #     quantized,indices = self.quantize_batch(x,self.values_per_latent)
    #     #print(f"xshape = {x.shape}, qshape= {quantized.shape}")
    #     # print(quantized_and_indices)
    #     # quantized = torch.stack([qi[0] for qi in quantized_and_indices])
    #     # indices = torch.stack([qi[1] for qi in quantized_and_indices])
    #     # print("quantized",quantized.shape,quantized.requires_grad)
    #     # print("x",x.shape,x.requires_grad , print(x.requires_grad))
    #     quantized_sg = x + (quantized - x).detach()
    #     # print("quantized_sg",quantized_sg.shape,quantized_sg.requires_grad)
    #     # assert False
    #     #quantized_sg = x + torch.autograd.Variable(quantized - x, requires_grad=False)
    #     outs = {
    #         'z_continuous': x,
    #         #'z_quantized':quantized_sg,
    #         'z_quantized': quantized,
    #         'z_hat': quantized_sg,
    #         'z_indices': indices
    #     }
    #     return outs

    # def sample(self):
    #     ret = []
    #     for values in self.values_per_latent:
    #         ret.append(torch.choice(values))
    #     return torch.tensor(ret)


import torch
import torch.nn as nn
import typing
import numpy as np

class StyleConvTranspose2DBlock( nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='leaky_relu', w_size=lsize):
        super(StyleConvTranspose2DBlock, self).__init__()
        
        self.forward_block = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conditioning = nn.Linear(w_size, 2 * out_channels)
        self.instance_norm = nn.GroupNorm(out_channels, out_channels, eps=1e-6)

        self.activation = self._get_activation(activation)
        
    def _get_activation(self, activation):
        if activation == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.2)
        elif activation == 'relu':
            return nn.ReLU()
        else:
            raise ValueError(f'unknown activation: {activation}')

    def forward(self, x, w):
        #cout = self.conditioning(w)
        #print(cout.shape,w.shape,x.shape)
        #print(x.shape
        #print(f"xin shape = {x.shape}")
        wout = self.conditioning(w)
        if wout.dim() == 1:
            wout = wout.view(1,-1)
        scale, bias = torch.chunk(wout, 2, dim=1)
        x = self.forward_block(x)
        #print(x.shape)
        x = self.instance_norm(x)
        #print(x.shape,scale.shape,bias.shape)
        #print("here")
        #print(f"xshape = {x.shape}, scale_shape = {scale.shape}, scale[:,none,none] shape = {scale[:,:,None,None].shape}")
        x = x * scale[:,:, None, None] + bias[:,:, None, None]
        x = self.activation(x)
        #assert False
        return x

class StyleConvTranspose2DBlocks( nn.Module):
    def __init__(self, widths, kernel_sizes, strides, paddings, activation, w_size = lsize ,fchannel=1):
        super(StyleConvTranspose2DBlocks, self).__init__()
        self.blocks = nn.ModuleList([
            StyleConvTranspose2DBlock(
                in_channels=widths[i-1] if i > 0 else fchannel,  # Assuming 1 for the input channel in the first block
                out_channels=width,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation=activation,
                w_size=w_size
            ) for i, (width, kernel_size, stride, padding) in enumerate(zip(widths, kernel_sizes, strides, paddings))
        ])

    def forward(self, x, w):
        for block in self.blocks:
            x = block(x, w)
        return x

import torch
import torch.nn as nn
import numpy as np
import jax
import jax.numpy as jnp
from disentangle.decoders import base

class StyleImageDecoder(nn.Module, base.Decoder):
    def __init__(self, dense_partial, style_conv_transpose_partial, conv_partial, input_channels =256 ,  out_shape = (1,3,64,64)):
        super(StyleImageDecoder, self).__init__()
        #input_shape = (input_channels,) + transition_shape[1:]
        input_shape = [1,256,4,4]
        dense_partial['in_size'] = lsize
        
        self.dense = initialize_model(DenseBlocks,dense_partial)#.to(device)
        #w = self.dense(torch.rand(1,10).to(device))
        self.input_map = 0.1 * torch.ones(input_shape, dtype=torch.float32).to(device)

        style_conv_transpose_partial['x'] = self.input_map
        style_conv_transpose_partial['w_size'] = 256# w.size()[1] #should be the size of the output of w.
        style_conv_transpose_partial['key'] = None
        style_conv_transpose_partial['fchannel'] = input_channels
        self.style_conv_transpose = initialize_model(StyleConvTranspose2DBlocks,style_conv_transpose_partial).to(device)
        #self.style_conv_transpose = style_conv_transpose_partial(x=self.input_map, w_size=w.shape[0], key=None)  # Replace with actual initialization
        
        #x = self.style_conv_transpose(x=self.input_map, w=w)

        conv_partial['in_channels']= 32#x.shape[1]
        conv_partial['out_channels'] = out_shape[1]
        conv_partial['key'] = None
        
        self.conv =  initialize_model(Conv2DBlock,conv_partial).to(device)  # Replace with actual initialization
        #x = self.conv(x)

        #assert x.shape == out_shape

    def forward(self, z):
        w = self.dense(z)
        x = self.style_conv_transpose(x=self.input_map.repeat(w.shape[0],1,1,1), w=w)
        outs = {
            'x_hat_logits':self.conv(x)
        }
        return outs
