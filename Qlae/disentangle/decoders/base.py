import torch
import torch.nn as nn
import abc

class Decoder( abc.ABC):

    @abc.abstractmethod
    def forward(self, x):
        pass
