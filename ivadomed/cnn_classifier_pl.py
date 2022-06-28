import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from models import CNNClassifier

# Problem parameters
dim = 1


class LitClassifier(pl.LightningModule):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        y_hat = self.classifier(x)

        return F.cross_entropy(y_hat.float(), y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


def get_transform(dim):
    if dim == 2:
        return transforms.ToTensor()
    elif dim == 1:
        def reduce(x: Tensor) -> Tensor:
            x = x.sum(1)
            x /= x.max()
            return x

        return transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(reduce)]
        )
    else:
        raise Exception("Unsupported number of dimensions.")


dataset = MNIST(os.getcwd(), download=True, transform=get_transform(dim))
train_loader = DataLoader(dataset, batch_size=8, num_workers=3)

# model
model_config_2d = {
    "CNN": {
        "filters": [8, 16, 32],
        "kernel_size": [3, 3],
        "activation": "leaky_relu",
        # "norm_layer": "batch",
        # "pool_layer": "max"
    },
    "MLP": {
        "neurons": [64, 32],
        "activation": "leaky_relu",
    },
    "num_classes": 10,
}

model_config_1d = {
    "CNN": {
        "filters": [8, 16, 32],
        "kernel_size": [3],
        "activation": "leaky_relu",
    },
    "MLP": {
        "neurons": [64, 32],
        "activation": "leaky_relu",
    },
    "num_classes": 10,
}
cnn_classfier = LitClassifier(CNNClassifier(eval(f"model_config_{dim}d")))

# train model
trainer = pl.Trainer()
trainer.fit(model=cnn_classfier, train_dataloaders=train_loader)
