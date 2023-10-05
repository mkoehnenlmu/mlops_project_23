import os
from contextlib import nullcontext as does_not_raise

import pytest
import torch
import yaml

from src.models.model import LightningModel
from src.models.train_model import load_data, normalize_data
from tests import _PROJECT_ROOT


def get_hparams():
    with open(os.path.join(_PROJECT_ROOT,
                           "src/configs/config.yaml"), "r") as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    hparams = {"lr": cfg["hyperparameters"]["learning_rate"],
               "epochs": cfg["hyperparameters"]["epochs"],
               "batch_size": cfg["hyperparameters"]["batch_size"],
               "input_size": cfg["hyperparameters"]["input_size"],
               "output_size": cfg["hyperparameters"]["output_size"],
               "hidden_size": cfg["hyperparameters"]["hidden_size"],
               "num_layers":  cfg["hyperparameters"]["num_layers"],
               "criterion":  cfg["hyperparameters"]["criterion"],
               "optimizer":  cfg["hyperparameters"]["optimizer"],
               }

    return hparams


def get_paths():
    with open(os.path.join(_PROJECT_ROOT,
                           "src/configs/config.yaml"), "r") as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    paths = {"training_data_path": cfg["paths"]["training_data_path"], }
    return paths


@pytest.mark.skipif(not os.path.exists(get_paths()["training_data_path"]),
                    reason="Data files not found")
def test_model_output():
    data = load_data(get_paths()["training_data_path"])
    x, _ = normalize_data(data)

    hparams = get_hparams()

    model = LightningModel(hparams)
    y_pred = model(x[:1])

    assert y_pred.shape[0] == hparams["output_size"], \
        "Model output has not the correct shape"


@pytest.mark.parametrize(
    "test_input,expectation",
    [(torch.randn(1, 90), does_not_raise()),
     (torch.randn(1, 2, 3), pytest.raises(
         ValueError,
         match='Expected input to a 2D tensor')),
     (torch.randn(20, 30), pytest.raises(
         ValueError,
         match=r'Expected each sample to have shape'
         + rf'\[{get_hparams()["input_size"]}\]')),
     ],)
def test_error_on_wrong_shape(test_input, expectation):
    hparams = get_hparams()
    model = LightningModel(hparams)

    with expectation:
        model(test_input)
