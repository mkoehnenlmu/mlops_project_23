import json
import os
import zipfile
from typing import Tuple

# import hydra
import pandas as pd
import torch
from google.cloud import storage
from torch import tensor

from src.models.model import LightningModel

from hydra import compose, initialize
# from omegaconf import OmegaConf

initialize(config_path="../configs/")
cfg = compose(config_name="config")

# load the paths into a global variablepaths
global paths
paths = cfg.paths
global add_configs
add_configs = cfg.hyperparameters


def get_paths():
    return dict(paths)


def get_additional_configs():
    return dict(add_configs)


# loads the data from the processed data folder
def load_data(
    data_path: str = paths.training_data_path, bucket_name: str = paths.training_bucket
) -> pd.DataFrame:
    """
    Loads training data from a CSV file.

    Args:
        data_path (str): Path to the training data CSV file.

    Returns:
        pd.DataFrame: Loaded training data as a DataFrame.
    """
    # if training data path is available
    if not os.path.exists(data_path):
        # pull the training data from google cloud storage
        data_path_for_download = data_path.split(".")[0] + ".zip"
        download_file_from_gcs(
            data_path_for_download.split("/")[1]
            + "/"
            + data_path_for_download.split("/")[2],
            data_path_for_download,
            bucket_name,
        )

        # unzip the file
        with zipfile.ZipFile(data_path, "r") as zip_ref:
            zip_ref.extractall(
                data_path.split("/")[0] + "/" + data_path.split("/")[1] + "/"
            )

    return pd.read_csv(data_path)


def load_model(model_path: str = paths.model_path) -> LightningModel:
    if not os.path.exists(model_path):
        download_model_from_gcs()
    hparams = get_hparams()
    # Load the model
    model = LightningModel(hparams=hparams)
    loaded_state_dict = torch.load(model_path)
    model.load_state_dict(loaded_state_dict)

    return model


def download_model_from_gcs(
    model_path: str = paths.model_path, bucket_name: str = paths.training_bucket
):
    # Download the model from Google Cloud Storage
    download_file_from_gcs(model_path, model_path, bucket_name)


def download_hparams_from_gcs(
    model_config_path: str = paths.model_config_path,
    bucket_name: str = paths.training_bucket,
):
    # Download the model from Google Cloud Storage
    download_file_from_gcs(
        model_config_path.split("/")[2] + "/" + model_config_path.split("/")[3],
        model_config_path,
        bucket_name,
    )


def get_hparams():
    if not os.path.exists(paths.model_config_path):
        download_model_from_gcs()
    with open(paths.model_config_path, "r") as json_file:
        hparams_str = json_file.read()
    hparams = json.loads(hparams_str)

    return hparams


# loads the data from the processed data folder
def download_file_from_gcs(
    gcs_data_path: str, local_data_path: str, bucket_name: str
) -> pd.DataFrame:
    """
    Loads training data from a CSV file.

    Args:
        data_path (str): Path to the training data CSV file.

    Returns:
        pd.DataFrame: Loaded training data as a DataFrame.
    """
    # on Cloud Compute Engine, the service account credentials
    # will be automatically available
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(
        gcs_data_path.split("/")[1] + gcs_data_path.split("/")[2]
    )  # "processed/train_sample.zip"

    # store the blob in training data path
    blob.download_to_filename(gcs_data_path)


def normalize_data(
    data: pd.DataFrame, dep_var: str = "DEP_DEL15"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize input data and split it into features (x) and targets (y).

    Args:
        data (pd.DataFrame): Input data in a DataFrame format.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing normalized features (x) and targets (y) as torch Tensors.
    """
    # convert data to tensors, where all columns in the dataframe
    # except TAc are inputs and TAc is the target
    x = tensor(data.drop(columns=[dep_var]).values).float()
    y = tensor(data[dep_var].values).float()

    # for every column in the input values, apply a min max normalization
    # that doesn't set any values to NaN
    for i in range(x.shape[1]):
        x[:, i] = (x[:, i] - x[:, i].min()) / (x[:, i].max() - x[:, i].min() + 1e-6)

    return x, y
