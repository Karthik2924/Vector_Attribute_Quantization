import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
import yaml
import inspect
from blocks import *
from infomec import*
import h5py
import torchvision
from fast_data import *

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

def transpose_and_gather(dict_list):
    # Transpose the list of dictionaries
    if len(dict_list) == 0:
        return {}

    result_dict = {}
    keys = dict_list[0].keys()

    for key in keys:
        # Check if the values corresponding to the key are dictionaries
        if all(isinstance(d[key], dict) for d in dict_list):
            # Recursively concatenate nested dictionaries
            result_dict[key] = transpose_and_gather([d[key] for d in dict_list])
        else:
            # Concatenate values if they are not dictionaries
            result_dict[key] = torch.cat([d[key] for d in dict_list]) if isinstance(dict_list[0][key], torch.Tensor) else [d[key] for d in dict_list]

    return result_dict


class falcor3d_data():
    def __init__(self, balanced_subset_size = 5000):
        super(falcor3d_data, self).__init__()
        self.subset_size = balanced_subset_size
        self.dir = '/mnt/beegfs/ksanka/data/falcor3d/Falcor3D_down128/images/'
        self.mul = np.array([46656,7776,1296,216,36,6,1])
        self.raw_labels = torch.from_numpy(np.load("/mnt/beegfs/ksanka/data/falcor3d/Falcor3D_down128/train-rec.labels"))
        self.size = self.raw_labels.shape[0]
        self.nlatents = self.raw_labels.shape[1]
        self.l = [5,6,6,6,6,6,6]
        self.cumulate = [0,5,11,17,23,29,35]
        #self.images = torch.from_numpy(np.load('/mnt/beegfs/ksanka/data/falcor3d/Falcor3D_down128/images.npy', mmap_mode='r+')).permute(0,3,1,2)/255.0
        self.images =np.load('/mnt/beegfs/ksanka/data/falcor3d/Falcor3D_down128/images.npy', mmap_mode='r+')
        self.get_variables()
    def get_variables(self):
        self.labels = (self.raw_labels*(torch.tensor(self.l)-1)).long()
        self.perm = torch.randperm(self.size)
        train_size = int(self.size*0.95)
        self.train_ind = self.perm[:train_size]
        self.train_size = self.train_ind.shape[0]
        self.test_ind = self.perm[train_size:]
        self.test_size = self.test_ind.shape[0]
        #self.sup,self.unsup,self.test = self.get_balanced_subset(self.subset_size)
    def get_train_batch(self,batch_size=64):
        ind = self.train_ind[torch.randint(0,self.train_size,(batch_size,))]
        return {'x':self.transform(self.images[ind]),'y':self.labels[ind],'z':self.raw_labels[ind]}
        #return self.images[ind],self.labels[ind],self.raw_labels[ind]
    def get_test_batch(self,batch_size=250):
        ind = self.test_ind[torch.randint(0,self.test_size,(batch_size,))]
        return {'x':self.transform(self.images[ind]),'y':self.labels[ind],'z':self.raw_labels[ind]}
    def transform(self,x):
        x = torch.from_numpy(x).permute(0,3,1,2)/255.0
        return torch.clamp(F.interpolate(x,size = (64,64),mode = 'bicubic'),min = 0,max =1)



def dist(x,v):
    return torch.abs(x.unsqueeze(1)-v)
class QuantizedLatent(nn.Module):
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
        if isinstance(num_values_per_latent, int):
            self.svpl = nn.Parameter(torch.stack( self._values_per_latent))
        else:
            mval = max(num_values_per_latent)
            vpl_padded = [torch.nn.functional.pad(tensor, (0, mval - tensor.size(0)),value = torch.inf) for tensor in self._values_per_latent]
            self.svpl = nn.Parameter(torch.stack(vpl_padded))

        self.optimize_values = optimize_values
        self.pquant = torch.vmap(self.quantize,1)


    @property
    def values_per_latent(self):
        if self.optimize_values:
            return self._values_per_latent
        else:
            return [torch.autograd.Variable(v, requires_grad=False) for v in self._values_per_latent]

    @staticmethod
    def quantize(x, values):
        distances = dist(x,values)
        index = torch.argmin(distances,1)
        return values[index], index


    def forward(self, x):
        #print(x.shape)
        # quantized_and_indices = [self.quantize(x_i, values_i) for x_i, values_i in zip(x, self.values_per_latent)]
        quantized,indices = self.pquant(x,self.svpl.T)
        #quantized_and_indices = [self.quantize(x_i, values_i) for x_i, values_i in zip(x.T, self.values_per_latent)]

        #quantized = torch.stack([qi[0] for qi in quantized_and_indices]).T
        #print("qshape",quantized.shape)
        #indices = torch.stack([qi[1] for qi in quantized_and_indices])
        
        #quantized_sg = x + torch.autograd.Variable(quantized.T - x, requires_grad=False)
        quantized_sg = x + (quantized.T - x).detach()
        outs = {
            'z_continuous': x,
            'z_quantized': quantized.T,
            'z_hat': quantized_sg,
            'z_indices': indices.T
        }

        return outs

    def sample(self):
        ret = []
        for values in self.values_per_latent:
            ret.append(torch.choice(values))
        return torch.tensor(ret)