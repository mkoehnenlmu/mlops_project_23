import pytorch_lightning as pl
from torch import save, tensor
import pandas as pd
from src.models.model import LightningModel
import hydra
import os

# loads the data from the processed data folder
def load_data(training_data_path: str):

    # if training data path is available
    if not os.path.exists(training_data_path):
        # pull the training data from google cloud storage

        from google.cloud import storage
        # on Cloud Run, the service account credentials will be automatically available
        storage_client = storage.Client()
        bucket = storage_client.get_bucket("delay_mlops_data")
        blob = bucket.blob("processed/data.csv")

        # store the blob in training data path
        blob.download_to_filename(training_data_path)
    return pd.read_csv(training_data_path)

def normalize_data(data: pd.DataFrame):
    # convert data to tensors, where all columns in the dataframe
    # except TAc are inputs and TAc is the target
    x = tensor(data.drop(columns=["TAc"]).values).float()
    y = tensor(data["TAc"].values).float()

    # for every column in the input values, apply a min max normalization
    for i in range(x.shape[1]):
        x[:,i] = (x[:,i] - x[:,i].min()) / (x[:,i].max() - x[:,i].min())

    return x,y

# trains the lightning model with the data
def train(data: pd.DataFrame, hparams: dict):
    # get data
    x,y = normalize_data(data)
    model = LightningModel(hparams)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="train_loss",
        mode="min",
        )

    #train the model with pytorch lightning and the hyperparameters
    trainer = pl.Trainer(max_epochs=hparams["epochs"],
                        gradient_clip_val=0.5,
                        limit_train_batches=30,
                        limit_val_batches=10,
                        logger=True,
                        callbacks=[checkpoint_callback],
                        )
    # train the model
    trainer.fit(model, model.train_dataloader(list(zip(x, y))))

    return model

# save the model in the models folder
def save_model(model: LightningModel):
    # save the trained model to the shared directory on disk
    save(model.state_dict(), "models/model.pth")

    # TODO: push the model to cloud storage with the tag "latest"


@hydra.main(config_path = "../configs/", config_name="config.yaml", version_base = "1.2")
def main(cfg):
    data = load_data(cfg.paths.training_data_path)
    
    hparams ={  "lr": cfg.hyperparameters.learning_rate,
                "epochs": cfg.hyperparameters.epochs,
                "batch_size": cfg.hyperparameters.batch_size,
                "input_size": cfg.hyperparameters.input_size,
                "output_size": cfg.hyperparameters.output_size,
                "hidden_size": cfg.hyperparameters.hidden_size,
                "num_layers":  cfg.hyperparameters.num_layers,
                "criterion":  cfg.hyperparameters.criterion,
                "optimizer":  cfg.hyperparameters.optimizer,
                }
    
    model = train(data, hparams)
    save_model(model)
    
    
if __name__ == "__main__":
    main()