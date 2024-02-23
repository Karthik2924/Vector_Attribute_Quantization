import torch
#from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from blocks import *
#from .types_ import *

class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim,img_size):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        #self.encoder = nn.Sequential([conv_block(3,16),conv_block(16,32),conv_block(32,64),conv_block(64,128)])
        self.encoder = nn.Sequential(conv_block(in_channels,8),conv_block(8,16),conv_block(16,32),conv_block(32,48),conv_block(48,60))


        self.fc_mu = nn.Linear(240, latent_dim)
        self.fc_var = nn.Linear(240, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 240)

        # self.l1 = nn.Linear(240,latent_dim)
        # self.l2 = nn.Linear(latent_dim,240)
        self.decoder = nn.Sequential(deconv_block(60,48),deconv_block(48,32),deconv_block(32,16),deconv_block(16,8),deconv_block(8,in_channels))
        if img_size == 128:
            self.encoder = nn.Sequential(conv_block(in_channels,8),conv_block(8,16),
                                         conv_block(16,32),conv_block(32,48),conv_block(48,60),conv_block(60,60))
            self.decoder = nn.Sequential(deconv_block(60,60),deconv_block(60,48),
                                         deconv_block(48,32),deconv_block(32,16),deconv_block(16,8),deconv_block(8,in_channels))

    def encode(self, input) :
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 60, 2, 2)
        result = self.decoder(result)
        #result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs) :
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) :
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input,reduction = 'sum')
        #recons_loss =F.binary_cross_entropy_with_logits(recons, input)
        #recons_loss =F.l1_loss(recons, input)



        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x, **kwargs) :
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    def get_latent(self,x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z
        