import os
import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from src.models.model import LightningModel
from torch import save, tensor, backends
from typing import Tuple, Dict, Any


# loads the data from the processed data folder
def load_data(training_data_path: str) -> pd.DataFrame:
    # if training data path is available
    if not os.path.exists(training_data_path):
        # pull the training data from google cloud storage

        from google.cloud import storage
        import zipfile

        # on Cloud Compute Engine, the service account credentials
        # will be automatically available
        storage_client = storage.Client()
        bucket = storage_client.get_bucket("delay_mlops_data")
        blob = bucket.blob("processed/train_sample.zip")

        # store the blob in training data path
        blob.download_to_filename("./data/processed/train_sample.zip")

        # unzip the file
        with zipfile.ZipFile(training_data_path, "r") as zip_ref:
            zip_ref.extractall("data/processed/")

    return pd.read_csv(training_data_path)


def normalize_data(data: pd.DataFrame, dep_var: str = "DEP_DEL15") -> Tuple[torch.Tensor, torch.Tensor]:
    # convert data to tensors, where all columns in the dataframe
    # except TAc are inputs and TAc is the target
    x = tensor(data.drop(columns=[dep_var]).values).float()
    y = tensor(data[dep_var].values).float()

    # for every column in the input values, apply a min max normalization
    # that doesn't set any values to NaN
    for i in range(x.shape[1]):
        x[:, i] = (x[:, i] - x[:, i].min()) / \
            (x[:, i].max() - x[:, i].min() + 1e-6)

    return x, y


# trains the lightning model with the data
def train(x: torch.Tensor, y: torch.Tensor, hparams: Dict[str, Any]):
    # if the loss function is SoftMarginLoss,
    # transform the target from {0,1} to {-1,1}
    if hparams["criterion"] == "SoftMarginLoss":
        y = y * 2 - 1

    hparams["input_size"] = x.shape[1]

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
        accelerator=hparams["device"],
    )
    # train the model

    trainer.fit(model, model.train_dataloader(list(zip(x, y.float()))))

    return model


def evaluate_model(model: LightningModel, data: pd.DataFrame):
    x, y = normalize_data(data)
    predictions = model.forward(x)
    # rmse = ((preds - y) ** 2).mean().sqrt()
    y = y.unsqueeze(1)

    # get the optimal threshold to maximize the f1 score
    best_threshold = 0.5
    f1 = 0
    for i in range(100):
        threshold = i / 100
        preds = (predictions >= threshold).float()
        true_pos = ((preds >= threshold) & (y == 1)).sum()
        false_pos = ((preds >= threshold) & (y == 0)).sum()
        false_neg = ((preds < threshold) & (y == 1)).sum()
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1_new = 2 * precision * recall / (precision + recall)
        if f1_new > f1:
            f1 = f1_new
            best_threshold = threshold

    # compute accuracy, precision, recall, and f1 score for preds and y
    accuracy = ((preds >= best_threshold) == y).sum() / y.shape[0]
    true_pos = ((preds >= best_threshold) == y).sum()
    false_pos = ((preds >= best_threshold) & (y == 0)).sum()
    false_neg = ((preds < best_threshold) & (y == 1)).sum()
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * precision * recall / (precision + recall)
    zero_one_loss = 1 - accuracy

    return accuracy, precision, recall, f1, zero_one_loss


# save the model in the models folder
def save_model(
    model: LightningModel, model_path: str, tag: str = "latest", push: bool = True
) -> None:
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
    x, y = normalize_data(data, "DEP_DEL15")
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)

    hparams = {
        "lr": cfg.hyperparameters.learning_rate,
        "epochs": cfg.hyperparameters.epochs,
        "batch_size": cfg.hyperparameters.batch_size,
        "input_size": cfg.hyperparameters.input_size,
        "output_size": cfg.hyperparameters.output_size,
        "hidden_size": cfg.hyperparameters.hidden_size,
        "hidden_layers": cfg.hyperparameters.hidden_layers,
        "criterion": cfg.hyperparameters.criterion,
        "optimizer": cfg.hyperparameters.optimizer,
        "dependent_var": cfg.hyperparameters.dependent_var,
        "device": cfg.hyperparameters.device,
    }

    if cfg.hyperparameters.device == "cpu":
        backends.cudnn.enabled = False

    model = train(x, y, hparams)

    acc, pred, recall, f1, zol = evaluate_model(model, test_data)

    print(
        "Model training loss (Accuracy, Precision, Recall, F1, 0-1 Loss): "
        + str(acc.item())
        + ", "
        + str(pred.item())
        + ", "
        + str(recall.item())
        + ", "
        + str(f1.item())
        + ", "
        + str(zol.item())
    )

    # TODO save scores with hyperparams in database

    save_model(model, push=False)
    # set tag as current timestamp
    tag = pd.Timestamp.now().strftime("%Y%m%d%H%M")
    save_model(model=model, model_path=cfg.paths.model_path, push=True, tag=tag)


if __name__ == "__main__":
    main()
