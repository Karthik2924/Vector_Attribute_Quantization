import torch
import torch.nn as nn
import torch.nn.functional as F

groups = 6
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens,groups = groups):
        super(Residual, self).__init__()
        self._block = nn.Sequential(

            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False,groups = groups),
            nn.GELU(),
            nn.GroupNorm(groups,num_residual_hiddens),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False,groups = groups),
            nn.GELU(),
            nn.GroupNorm(groups, in_channels),
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,groups = 1):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens,groups = groups)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.gelu(x)

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()
        self._block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels//2,
                      kernel_size=(3,3), stride=(2,2),padding = 1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(out_channels//2),
        )
        self._block2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels//2,
                        out_channels=out_channels,
                        kernel_size=(3,3), stride=(1,1),padding = 1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        z = self._block(x)
        #print(z.shape)
        return self._block2(z)

class deconv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(deconv_block, self).__init__()
        self._block = nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, 
                            out_channels=out_channels*2,
                            kernel_size=4, 
                            stride=2, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(out_channels*2),
        )
        self._block2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*2,
                        out_channels=out_channels,
                        kernel_size=(3,3), stride=(1,1),padding = 1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(out_channels)
        )
        
        
    def forward(self, x):
        z = self._block(x)
        #print(z.shape)
        return self._block2(z)

class group_conv_block(nn.Module):
    def __init__(self, in_channels, out_channels,stride,kernel,padding,groups):
        super(group_conv_block, self).__init__()
        self._block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=kernel, stride=stride,padding = padding, bias=True,groups = groups),
            nn.GELU(),
            nn.GroupNorm(groups,out_channels)
        )
        #print(out_channels,groups)
        
    def forward(self, x):
        z = self._block(x)
        return z
        #print(z.shape)
        return self._block2(z)