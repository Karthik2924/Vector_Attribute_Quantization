import torch
import torch.nn as nn
import abc

class Block(nn.Module, abc.ABC):

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass
