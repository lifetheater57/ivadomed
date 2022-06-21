import os

import pytorch_lightning as pl
import torch
from torch import nn, Tensor
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from typing import Type, Any, Callable, Union, List, Optional

from utils.model import outSizeCNN, genReLuCNN, genReLuCNNTranpose
from losses import KullbackLeiblerLoss

def conv5x1(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv1d:
    """5x1 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)

class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer:
            self.norm_layer = norm_layer
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv5x1(inplanes, planes, stride)
        if norm_layer:
            self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x1(planes, planes)
        if norm_layer:
            self.bn2 = norm_layer(planes)
        self.conv3 = conv5x1(planes, planes)
        if norm_layer:
            self.bn3 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        if self.norm_layer:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.norm_layer:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.norm_layer:
            out = self.bn3(out)
        out = self.relu(out)

        if self.downsample:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    def __init__(self, filters, obs_channels, kernel, stride, flattened_dim, latent_dim):
        super().__init__()
        # Note: kernel is currently unused
        # Creation of the encoder's CNN
        CNN_encoder = nn.Sequential()
        for i in range(len(filters)):
            in_channels = filters[i - 1] if i > 0 else obs_channels
            out_channels = filters[i]
            #TODO: add option to use BasicBlock
            module = genReLuCNN(in_channels, out_channels, kernel, stride)
            module_name = "enc_conv_relu" + str(i)

            CNN_encoder.add_module(module_name, module)

        # Initialization of the layer on top of the CNN of the encoder 
        # and its weights and biases
        #TODO: add parameter list to set mlp layer sizes
        encoder_linear_layer = nn.Linear(flattened_dim, 64)
        nn.init.kaiming_normal_(encoder_linear_layer.weight, a=0.01, nonlinearity="leaky_relu")

        # Creation of the encoder
        #TODO: rename because encoder include the generation of mu and sigma
        self.encoder = nn.Sequential(
            CNN_encoder, 
            nn.Flatten(), 
            encoder_linear_layer,
            #nn.BatchNorm1d(64),
            nn.LeakyReLU()
        )

        # Creation of the latent space mean and variance layers
        self.mu = nn.Sequential(nn.Linear(64, latent_dim))
        self.log_var = nn.Sequential(nn.Linear(64, latent_dim))

    def forward(self, x):
        # Encode the example
        x = self.encoder(x)        
        # Get mean and variance of the latent variables for the example
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, filters, obs_channels, kernel, stride, flattened_dim, latent_dim, out_dims):
        super().__init__()
        #TODO: add sampling for output dimensions
        # Creation of the decoder's CNN
        CNN_decoder = nn.Sequential()
        for i in reversed(range(len(filters))):
            in_channels = filters[i]
            out_channels = filters[i - 1] if i > 0 else obs_channels#2 * obs_channels

            out_size = outSizeCNN(
                out_dims[i + 1], kernel, stride, transposed=True
            )[1]
            output_padding = tuple(out_dims[i] - out_size)
            #TODO: change this CNN by the TransposedBasicBlock
            module = genReLuCNNTranpose(
                in_channels,
                out_channels,
                kernel,
                stride,
                output_padding=output_padding
            )
            module_name = "dec_relu_conv" + str(len(filters) - i - 1)

            CNN_decoder.add_module(module_name, module)

        # Initialization of the layer on top of the CNN of the decoder 
        # and its weights and biases
        decoder_linear_layer = nn.Linear(latent_dim, 64)
        nn.init.kaiming_normal_(decoder_linear_layer.weight, a=0.01, nonlinearity="leaky_relu")
        
        # Creation of the decoder 
        self.decoder = nn.Sequential(
            decoder_linear_layer,
            #nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, flattened_dim),
            nn.Unflatten(1, (filters[-1], int(out_dims[-1, 0]), int(out_dims[-1, 1]))),
            CNN_decoder,
        )

    def forward(self, x):
        return self.decoder(x)


class VAE(nn.Module):
    #TODO: check when to disconnect the gradient
    def __init__(self, shape, latent_dim=32):
        super().__init__()

        # Splitting shape
        obs_channels, obs_height, obs_width = shape

        # Initializing constant params
        kernel = 4
        stride = 2
        filters = [32, 64]#, 128]

        # Computing the dims required by the flattening and unflattening ops
        in_dims = np.array([obs_height, obs_width])
        out_dims = outSizeCNN(in_dims, kernel, stride, len(filters))
        flattened_dims = filters[-1] * out_dims[-1, 0] * out_dims[-1, 1]

        self.encoder = Encoder(filters, obs_channels, kernel, stride, flattened_dims, latent_dim)
        self.decoder = Decoder(filters, obs_channels, kernel, stride, flattened_dims, latent_dim, out_dims)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        # Sample from the latent space
        z = self.sample_latent_space(mu, log_var)        
        # Decode the sample
        x_hat = self.decoder(z)

        return x_hat, mu, log_var

    def sample_latent_space(self, mu, log_var):
        return mu + torch.mul(torch.exp(log_var / 2.0), torch.randn_like(log_var))

    def encode(self, x, sample=True):
        # Get mean and variance of the latent variables from the encoded example
        mu, log_var = self.encoder(x)

        if sample:
            # Sample from the latent space
            z = self.sample_latent_space(mu, log_var)
            return z
        else:
            return mu, log_var

    def decode(self, z, sample=False):
        # Decode the sample
        decoded = self.decoder(z)
        if sample:
            """# Get mean and variance of the output values from the decoded sample
            mus = decoded[:, 0::2, :, :]
            log_vars = decoded[:, 1::2, :, :]
            # Sample from the parameters
            sampled = self.sample_latent_space(mus, log_vars)
            return sampled"""
            pass
        else:
            return decoded
