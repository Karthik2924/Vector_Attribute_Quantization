import torch
import torch.nn as nn
import abc
class Encoder(nn.Module,abc.ABC):
    def __init__(self, transition_shape = (3,64,64)):
        super(Encoder, self).__init__()
        self.transition_shape = transition_shape
    
    @abc.abstractmethod
    def forward(self, x):
        pass
        #raise NotImplementedError("Subclasses must implement the forward method.")
