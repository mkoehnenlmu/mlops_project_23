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
            nn.Linear(int(hparams["hidden_size"]),
                      int(hparams["hidden_size"]/2)),
            nn.ReLU(),
            nn.Linear(int(hparams["hidden_size"]/2),
                      int(hparams["hidden_size"]/4)),
            nn.ReLU(),
            nn.Linear(int(hparams["hidden_size"]/4), hparams["output_size"]),
        )

        # use l2 loss
        if hparams["criterion"] == "MSELoss":
            self.loss = nn.MSELoss()
        elif hparams["criterion"] == "NLLLoss":
            self.loss = nn.NLLLoss()
        else:
            raise NotImplementedError

    def train_dataloader(self, data):
        return DataLoader(data, batch_size=self.hyperparams["batch_size"],
                          shuffle=True)

    def configure_optimizers(self):
        if self.hyperparams["optimizer"] == "Adam":
            return optim.Adam(self.parameters(), lr=self.hyperparams["lr"])
        elif self.hyperparams["optimizer"] == "SGD":
            return optim.SGD(self.parameters(), lr=self.hyperparams["lr"])
        elif self.hyperparams["optimizer"] == "AdamW":
            return optim.AdamW(self.parameters(), lr=self.hyperparams["lr"])
        else:
            raise NotImplementedError

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError('Expected input to a 2D tensor')
        if x.shape[1] != self.hyperparams["input_size"]:
            raise ValueError('Expected each sample to have shape'
                             + f'[{self.hyperparams["input_size"]}]')
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)

        return {'loss': loss}
