import os

import pytorch_lightning as pl
import torch
from torch import nn, Tensor
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from typing import Type, Any, Callable, Union, List, Optional
from models import VAE

from losses import KullbackLeiblerLoss


class LitBetaVAE(pl.LightningModule):
    def __init__(self, vae, beta):
        super().__init__()
        self.vae = vae
        self.beta = beta

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x_hat, mu, log_var = self.vae(x)

        reconstruction_loss = torch.pow(x - x_hat, 2).mean() * x[0].nelement()
        kl_loss = KullbackLeiblerLoss()(mu, log_var)
        return reconstruction_loss + self.beta * kl_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset, batch_size=8, num_workers=3)

# model
model_config_2d = {
    "shape": [1, 28, 28],
    "encoder": {
        "CNN": {
            "filters": [8, 16],
            "kernel_size": [3, 3],
            "activation": "leaky_relu",
            "norm_layer": "batch",
            # "pool_layer": "max"
        },
        "MLP": {
            "neurons": [32],
            "activation": "leaky_relu",
        },
        "num_classes": 10,
    },
}
autoencoder = LitBetaVAE(VAE(model_config_2d), 1.0)

# train model
trainer = pl.Trainer()
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
