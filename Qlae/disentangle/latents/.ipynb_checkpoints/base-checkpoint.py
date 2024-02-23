import abc
import torch
import torch.nn as nn



class Latent( abc.ABC):
    is_continuous: True
    num_latents: 12
    num_inputs: 10


    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass
