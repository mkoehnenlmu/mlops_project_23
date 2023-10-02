import pytorch_lightning as pl
from torch import save, tensor
import pandas as pd
from model import LightningModel


# loads the data from the processed data folder
def load_data():
    return pd.read_csv('data/processed/data.csv')


# trains the lightning model with the data
def train(data: pd.DataFrame, hparams: dict):
    # convert data to tensors, where all columns in the dataframe
    # except TAc are inputs and TAc is the target
    x = tensor(data.drop(columns=["TAc"]).values).float()
    y = tensor(data["TAc"].values).float()

    # for every column in the input values, apply a min max normalization
    for i in range(x.shape[1]):
        x[:,i] = (x[:,i] - x[:,i].min()) / (x[:,i].max() - x[:,i].min())

    model = LightningModel(hparams)

    #train the model with pytorch lightning and the hyperparameters
    trainer = pl.Trainer(max_epochs=hparams["epochs"],
                        gradient_clip_val=0.5,
                        limit_train_batches=30,
                        limit_val_batches=10,
                        logger=True,
                        )
    # train the model
    trainer.fit(model, model.train_dataloader(list(zip(x, y))))

    return model

# save the model in the models folder
def save_model(model: LightningModel):
    # save the trained model to the shared directory on disk
    save(model.state_dict(), "models/model.pth")


if __name__ == "__main__":
    data = load_data()

    hparams ={  "lr": 0.001,
                "epochs": 5,
                "batch_size": 32,
                "input_size": 90,
                "hidden_size": 128,
                "num_layers": 3,
                }
    
    model = train(data, hparams)
    save_model(model)