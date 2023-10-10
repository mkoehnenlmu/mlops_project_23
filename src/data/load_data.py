import os
from typing import Tuple

import pandas as pd
import torch
from torch import tensor


# loads the data from the processed data folder
def load_data(training_data_path: str) -> pd.DataFrame:
    """
    Loads training data from a CSV file.

    Args:
        training_data_path (str): Path to the training data CSV file.

    Returns:
        pd.DataFrame: Loaded training data as a DataFrame.
    """
    # if training data path is available
    if not os.path.exists(training_data_path):
        # pull the training data from google cloud storage

        import zipfile

        from google.cloud import storage

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
        x[:, i] = (x[:, i] - x[:, i].min()) / \
            (x[:, i].max() - x[:, i].min() + 1e-6)

    return x, y
