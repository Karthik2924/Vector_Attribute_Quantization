import abc
import typing
import torch
import torch.nn as nn
import torch.optim as optim

import disentangle


class Model(nn.Module, abc.ABC):
    # encoder: disentangle.encoders.Encoder
    # latent: disentangle.latents.Latent
    # decoder: disentangle.decoders.Decoder
    # lambdas: typing.Mapping[str, float]
    inference: bool

    def __init__(self):
        super(Model, self).__init__()
        self.inference = False

    def train(self):
        self.inference = False

    def eval(self):
        self.inference = True

    def filter(self, x=None):
        if x is None:
            return [p for p in self.parameters() if p.requires_grad]
        else:
            return [p for p in x.parameters() if p.requires_grad]

    @abc.abstractmethod
    def construct_optimizer(self, config):
        pass

    @abc.abstractmethod
    def param_labels(self):
        pass
