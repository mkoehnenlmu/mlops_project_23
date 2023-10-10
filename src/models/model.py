from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset


class LightningModel(pl.LightningModule):
    def __init__(self, hparams: Dict[str, Any]):
        super(LightningModel, self).__init__()

        self.hyperparams = hparams

        # define the neural network
        layers = []
        layers.append(nn.Linear(hparams["input_size"], int(hparams["hidden_size"])))
        # add activation
        layers.append(nn.ReLU())
        # layers.append(
        #    nn.Linear(int(hparams["hidden_size"] / 2), hparams["hidden_size"])
        # )
        for i in range(hparams["hidden_layers"]):
            layers.append(
                nn.Linear(
                    int(hparams["hidden_size"] / (i + 1)),
                    int(hparams["hidden_size"] / (i + 2)),
                )
            )
            layers.append(nn.ReLU())
        layers.append(
            nn.Linear(
                int(hparams["hidden_size"] / (hparams["hidden_layers"] + 1)),
                hparams["output_size"],
            )
        )
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

        if hparams["criterion"] == "MSELoss":
            self.loss = nn.MSELoss()
        elif hparams["criterion"] == "NLLLoss":
            self.loss = nn.NLLLoss()
        elif hparams["criterion"] == "HuberLoss":
            self.loss = nn.HuberLoss()
        elif hparams["criterion"] == "BCELoss":
            self.loss = nn.BCELoss()
        elif hparams["criterion"] == "SoftMarginLoss":
            self.loss = nn.SoftMarginLoss()
        else:
            raise NotImplementedError

    def train_dataloader(self, data: Dataset) -> DataLoader:
        return DataLoader(
            dataset=data, batch_size=self.hyperparams["batch_size"], shuffle=True
        )

    def configure_optimizers(self) -> optim.Optimizer:
        if self.hyperparams["optimizer"] == "Adam":
            return optim.Adam(self.parameters(), lr=self.hyperparams["lr"])
        elif self.hyperparams["optimizer"] == "SGD":
            return optim.SGD(self.parameters(), lr=self.hyperparams["lr"])
        elif self.hyperparams["optimizer"] == "AdamW":
            return optim.AdamW(self.parameters(), lr=self.hyperparams["lr"])
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("Expected input to be a 2D tensor")
        if x.shape[1] != self.hyperparams["input_size"]:
            raise ValueError(
                "Expected each sample to have shape"
                + f'[{self.hyperparams["input_size"]}]'
            )
        if self.hyperparams["criterion"] == "SoftMarginLoss":
            return self.model(x).clamp(min=-1, max=1)
        else:
            return self.model(x).clamp(min=0, max=1)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y.unsqueeze(1))
        self.log("train_loss", loss)

        return loss
