import os

import pytorch_lightning as pl
import torch
from torch import nn, Tensor
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from typing import Type, Any, Callable, Union, List, Optional
from models import CNN, MLP, instantiate_config

from utils.model import outSizeCNN
from losses import KullbackLeiblerLoss


def conv5x1(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv1d:
    """5x1 convolution with padding"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=5,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=True,
        dilation=dilation,
    )


class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
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
    def __init__(self, config):
        super().__init__()
        # TODO: validate CNN and MLP config once loaded
        config_cnn = config.get("CNN", None)
        config_mlp = config.get("MLP", None)
        latent_dim = config.get("latent_dim", 2)

        # Instantiate activation, normalization, pooling
        config_cnn = instantiate_config(config_cnn)
        config_mlp = instantiate_config(config_mlp)

        modules = []
        if config_cnn is not None:
            modules.append(CNN(**config_cnn))
        modules.append(nn.Flatten())
        if config_mlp is not None:
            modules.append(MLP(**config_mlp))
        self.encoder = nn.Sequential(*modules)

        # Creation of the latent space mean and variance layers
        self.mu = nn.Sequential(nn.LazyLinear(latent_dim))
        self.log_var = nn.Sequential(nn.LazyLinear(latent_dim))

    def forward(self, x):
        # Encode the example
        x = self.encoder(x)
        # Get mean and variance of the latent variables for the example
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, config, shape, decoder_config=False):
        super().__init__()
        # TODO: add sampling for output dimensions

        # TODO: validate CNN and MLP config once loaded
        config_cnn = config.get("CNN", None)
        config_mlp = config.get("MLP", None)

        # Change CNN and MLP config to be reversed encoder
        if not decoder_config:
            # CNN
            if config_cnn:
                filters = config_cnn["filters"]
                kernel = config_cnn["kernel_size"]
                stride = config_cnn.get("stride", 1)
                cnn_out_dims = outSizeCNN(shape[1:], kernel, stride, n=len(filters))
                target_ctnn_out_dims = np.flip(cnn_out_dims, 0)
                ctnn_out_dims = np.apply_along_axis(
                    outSizeCNN,
                    1,
                    target_ctnn_out_dims,
                    kernel,
                    stride,
                    n=1,
                    transpose=True,
                )[:, 1]
                padding = list(
                    map(tuple, target_ctnn_out_dims[1:] - ctnn_out_dims[:-1])
                )
                preflattened_dim = (
                    filters[-1],
                    int(cnn_out_dims[-1, 0]),
                    int(cnn_out_dims[-1, 1]),
                )
                flattened_dim = filters[-1] * np.prod(cnn_out_dims[-1])

                # Update CNN config
                config_cnn["filters"] = reversed(config_cnn["filters"])
                config_cnn["padding"] = padding
                config_cnn["transpose"] = True
                if config_cnn.get("pool_every", None):
                    config_cnn["pool_offset"] = len(filters) % config_cnn["pool_every"]
            else:
                preflattened_dim = shape
                flattened_dim = np.prod(np.asarray(shape))

            # Update MLP config
            if config_mlp:
                config_mlp["neurons"] = reversed(config_mlp["neurons"])

        # Instantiate activation, normalization, pooling
        config_cnn = instantiate_config(config_cnn)
        config_mlp = instantiate_config(config_mlp)

        modules = []
        if config_mlp is not None:
            modules.append(MLP(**config_mlp))
        modules.append(nn.LazyLinear(flattened_dim))
        modules.append(nn.Unflatten(1, preflattened_dim))
        if config_cnn is not None:
            modules.append(CNN(**config_cnn))
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.decoder(x)


class VAE(nn.Module):
    # TODO: check when to disconnect the gradient
    def __init__(self, config):
        super().__init__()

        # Computing the dims required by the flattening and unflattening ops
        self.encoder = Encoder(config["encoder"])
        self.decoder = Decoder(config["encoder"], config["shape"])

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
