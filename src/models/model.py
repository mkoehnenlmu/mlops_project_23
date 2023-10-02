# define a lightning model

# Path: src/models/model.py
import pytorch_lightning as pl
import torch.nn as nn
from torch import optim

from torch.utils.data import DataLoader

class LightningModel(pl.LightningModule):

    def __init__(self, hparams):
        super(LightningModel, self).__init__()

        self.hyperparams = hparams

        # one input layer with inputs determined by the input_size parameter
        # hidden layers determined by the hidden_size parameter
        # one output layer with one output
        self.model = nn.Sequential(
            nn.Linear(hparams["input_size"], hparams["hidden_size"]),
            nn.ReLU(),
            nn.Linear(int(hparams["hidden_size"]), int(hparams["hidden_size"]/2)),
            nn.ReLU(),
            nn.Linear(int(hparams["hidden_size"]/2), int(hparams["hidden_size"]/4)),
            nn.ReLU(),
            nn.Linear(int(hparams["hidden_size"]/4), 1),
        )

        # use l2 loss
        self.loss = nn.MSELoss()
    
    def train_dataloader(self, data):
        return DataLoader(data, batch_size=self.hyperparams["batch_size"],
                          shuffle=True)
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hyperparams["lr"])

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return {'loss': loss}