import os

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from src.models.model import LightningModel
from torch import save, tensor
from typing import Tuple


# loads the data from the processed data folder
def load_data(training_data_path: str) -> pd.DataFrame:
    # if training data path is available
    if not os.path.exists(training_data_path):
        # pull the training data from google cloud storage

        from google.cloud import storage

        # on Cloud Compute Engine, the service account credentials
        # will be automatically available
        storage_client = storage.Client()
        bucket = storage_client.get_bucket("delay_mlops_data")
        blob = bucket.blob("processed/data.csv")

        # store the blob in training data path
        blob.download_to_filename(training_data_path)
    return pd.read_csv(training_data_path)


def normalize_data(data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    # convert data to tensors, where all columns in the dataframe
    # except TAc are inputs and TAc is the target
    x = tensor(data.drop(columns=["TAc"]).values).float()
    y = tensor(data["TAc"].values).float()

    # for every column in the input values, apply a min max normalization
    for i in range(x.shape[1]):
        x[:, i] = (x[:, i] - x[:, i].min()) / (x[:, i].max() - x[:, i].min())

    return x, y


# trains the lightning model with the data
def train(x: torch.Tensor, y: torch.Tensor, hparams: dict) -> LightningModel:
    model = LightningModel(hparams)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="train_loss",
        mode="min",
    )

    # train the model with pytorch lightning and the hyperparameters
    trainer = pl.Trainer(
        max_epochs=hparams["epochs"],
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
def save_model(model: LightningModel, model_path: str, tag: str = "latest", push: bool = True):
    # save the trained model to the shared directory on disk
    save(model.state_dict(), model_path)

    # push the model to google cloud storage
    if push:
        from google.cloud import storage

        # on Cloud Compute Engine, the service account credentials
        # will be automatically available
        storage_client = storage.Client()
        bucket = storage_client.get_bucket("delay_mlops_data")
        # upload the trained model to the bucket with
        # the tag set
        blob = bucket.blob(model_path)
        blob.upload_from_filename(model_path)


@hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def main(cfg):
    # get data
    data = load_data(cfg.paths.training_data_path)
    x, y = normalize_data(data)

    hparams = {
        "lr": cfg.hyperparameters.learning_rate,
        "epochs": cfg.hyperparameters.epochs,
        "batch_size": cfg.hyperparameters.batch_size,
        "input_size": cfg.hyperparameters.input_size,
        "output_size": cfg.hyperparameters.output_size,
        "hidden_size": cfg.hyperparameters.hidden_size,
        "num_layers": cfg.hyperparameters.num_layers,
        "criterion": cfg.hyperparameters.criterion,
        "optimizer": cfg.hyperparameters.optimizer,
    }

    model = train(x, y, hparams)
    # set tag as current timestamp
    tag = pd.Timestamp.now().strftime("%Y%m%d%H%M")
    save_model(model=model, model_path=cfg.paths.model_path, push=True, tag=tag)


if __name__ == "__main__":
    main()
