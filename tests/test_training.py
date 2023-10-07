import yaml
from src.models.model import LightningModel
import torch
from src.models.train_model import train
from tests import _PROJECT_ROOT
import os

from tests.define_test_data import get_normalized_test_data


def get_test_hparams():
    with open(os.path.join(_PROJECT_ROOT, "src/configs/config.yaml"), "r") as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    hparams = {
        "lr": cfg["hyperparameters"]["learning_rate"],
        "epochs": 1,
        "batch_size": cfg["hyperparameters"]["batch_size"],
        "input_size": cfg["hyperparameters"]["input_size"],
        "output_size": cfg["hyperparameters"]["output_size"],
        "hidden_size": cfg["hyperparameters"]["hidden_size"],
        "num_layers": cfg["hyperparameters"]["num_layers"],
        "criterion": cfg["hyperparameters"]["criterion"],
        "optimizer": cfg["hyperparameters"]["optimizer"],
    }
    return hparams


def test_model_created():
    x, y = get_normalized_test_data()
    hparams = get_test_hparams()
    # Call the train function with sample data and hyperparameters
    model = train(x, y, hparams)

    # Assert that the returned model is not None
    assert model is not None, "Model not instantiated."
    assert isinstance(model, LightningModel), "Model not instance of specified model."


def test_model_hyperparameters():
    x, y = get_normalized_test_data()
    hparams = get_test_hparams()
    # Call the train function with sample data and hyperparameters
    model = train(x, y, hparams)

    # Assert hyperparameters
    if hparams["criterion"] == "MSELoss":
        assert isinstance(model.loss, torch.nn.MSELoss), "Incorrect loss used."
    elif hparams["criterion"] == "NLLLoss":
        assert isinstance(model.loss, torch.nn.NLLLoss), "Incorrect loss used."
    if hparams["optimizer"] == "Adam":
        assert isinstance(model.configure_optimizers(), torch.optim.Adam), "Incorrect optimizer used."
    elif hparams["optimizer"] == "SGD":
        assert isinstance(model.configure_optimizers(), torch.optim.SGD), "Incorrect optimizer used."


def test_model_logs():
    x, y = get_normalized_test_data()
    hparams = get_test_hparams()

    # Call the train function with sample data and hyperparameters
    model = train(x, y, hparams)

    # Assert logged loss
    assert model.logger.log_dir is not None, "Train loss has not been logged."
