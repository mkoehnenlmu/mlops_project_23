import yaml
from src.models.model import LightningModel
import torch
from src.models.train_model import load_data
from src.models.train_model import train
from tests import _PROJECT_ROOT
import os
import pytest


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
def test_model_created():
    data = load_data(get_paths()["training_data_path"])
    sample_data = data[:5]
    hparams = get_hparams()

    # Call the train function with sample data and hyperparameters
    model = train(sample_data, hparams)

    # Assert that the returned model is not None
    assert model is not None, "Model not instantiated."
    assert isinstance(model, LightningModel), \
        "Model not instance of specified model."


@pytest.mark.skipif(not os.path.exists(get_paths()["training_data_path"]),
                    reason="Data files not found")
def test_model_hyperparameters():
    data = load_data(get_paths()["training_data_path"])
    sample_data = data[:5]
    hparams = get_hparams()
    print(hparams["lr"])
    # Call the train function with sample data and hyperparameters
    model = train(sample_data, hparams)

    # Assert hyperparameters
    if hparams["criterion"] == "MSELoss":
        assert isinstance(model.loss, torch.nn.MSELoss), "Incorrect loss used."
    elif hparams["criterion"] == "NLLLoss":
        assert isinstance(model.loss, torch.nn.NLLLoss), "Incorrect loss used."
    if hparams["optimizer"] == "Adam":
        assert isinstance(model.configure_optimizers(), torch.optim.Adam), \
            "Incorrect optimizer used."
    elif hparams["optimizer"] == "SGD":
        assert isinstance(model.configure_optimizers(), torch.optim.SGD), \
            "Incorrect optimizer used."


@pytest.mark.skipif(not os.path.exists(get_paths()["training_data_path"]),
                    reason="Data files not found")
def test_model_logs():
    data = load_data(get_paths()["training_data_path"])
    sample_data = data[:5]
    hparams = get_hparams()

    # Call the train function with sample data and hyperparameters
    model = train(sample_data, hparams)

    # Assert logged loss
    assert model.logger.log_dir is not None, "Train loss has not been logged."
