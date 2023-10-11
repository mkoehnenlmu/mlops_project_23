from typing import Any, Dict, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import save

from src.data.load_data import create_normalized_target
from src.models.model import LightningModel


# trains the lightning model with the data
def train_logged(
    x: torch.Tensor,
    y: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    hparams: Dict[str, Any],
) -> LightningModel:
    """
    Train a LightningModel using the given data and hyperparameters.

    Args:
        x (torch.Tensor): Input features as a torch Tensor.
        y (torch.Tensor): Target values as a torch Tensor.
        x_test (torch.Tensor): Input features as a torch Tensor.
        y_test (torch.Tensor): Target values as a torch Tensor.
        hparams (Dict[str, Any]): Hyperparameters for training.

    Returns:
        LightningModel: Trained LightningModel.
    """
    # if the loss function is SoftMarginLoss,
    # transform the target from {0,1} to {-1,1}
    if hparams["criterion"] == "SoftMarginLoss":
        y = y * 2 - 1
        y_test = y_test * 2 - 1

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
        logger=pl.loggers.TensorBoardLogger(
            save_dir="logs/",
            name="lightning_logs",
            version="0",
        ),
        callbacks=[checkpoint_callback],
        accelerator=hparams["device"],
    )

    trainer.logger.log_hyperparams(hparams)

    trainer.fit(
        model=model,
        train_dataloaders=model.train_dataloader(list(zip(x, y.float()))),
        val_dataloaders=model.val_dataloader(list(zip(x_test, y_test.float()))),
    )

    return model


def train(x: torch.Tensor, y: torch.Tensor, hparams: Dict[str, Any]) -> LightningModel:
    """
    Train a LightningModel using the given data and hyperparameters.

    Args:
        x (torch.Tensor): Input features as a torch Tensor.
        y (torch.Tensor): Target values as a torch Tensor.
        hparams (Dict[str, Any]): Hyperparameters for training.

    Returns:
        LightningModel: Trained LightningModel.
    """
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
        limit_val_batches=0,
        logger=True,
        callbacks=[checkpoint_callback],
        accelerator=hparams["device"],
        num_sanity_val_steps=0,
    )

    # train the model without validation data
    trainer.fit(model, model.train_dataloader(list(zip(x, y.float()))))

    return model


def evaluate_model(
    model: LightningModel, data: pd.DataFrame
) -> Tuple[float, float, float, float, float]:
    x, y = create_normalized_target(data)
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

    return (
        accuracy.item(),
        precision.item(),
        recall.item(),
        f1.item(),
        zero_one_loss.item(),
    )


# save the model in the models folder
def save_model(
    model: LightningModel, model_path: str, bucket_name: str, push: bool = True
) -> None:
    """
    Save a trained model to a file and optionally push it to Google Cloud Storage.

    Args:
        model (LightningModel): Trained LightningModel to be saved.
        model_path (str): Path to save the model.
        tag (str, optional): Tag to be added to the model file name. Defaults to "latest".
        push (bool, optional): Whether to push the model to Google Cloud Storage. Defaults to True.

    Returns:
        None
    """
    # save the trained model to the shared directory on disk
    save(model.state_dict(), model_path)

    # push the model to google cloud storage
    if push:
        from google.cloud import storage

        # on Cloud Compute Engine, the service account credentials
        # will be automatically available
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        # upload the trained model to the bucket with
        # the tag set
        blob = bucket.blob(model_path)
        blob.upload_from_filename(model_path)
