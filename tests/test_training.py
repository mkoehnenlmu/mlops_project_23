import os

import torch

from src.models.model import LightningModel
from src.models.train import train, save_model
from tests.utilities import get_test_hparams, get_normalized_test_data, get_paths


def test_model_created() -> None:
    x, y = get_normalized_test_data()
    hparams = get_test_hparams()
    # Call the train function with sample data and hyperparameters
    model = train(x, y, hparams)

    # Assert that the returned model is not None
    assert model is not None, "Model not instantiated."
    assert isinstance(model, LightningModel), "Model not instance of specified model."


def test_model_hyperparameters() -> None:
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
        assert isinstance(
            model.configure_optimizers(), torch.optim.Adam
        ), "Incorrect optimizer used."
    if hparams["optimizer"] == "AdamW":
        assert isinstance(
            model.configure_optimizers(), torch.optim.AdamW
        ), "Incorrect optimizer used."
    elif hparams["optimizer"] == "SGD":
        assert isinstance(
            model.configure_optimizers(), torch.optim.SGD
        ), "Incorrect optimizer used."


def test_model_logs() -> None:
    x, y = get_normalized_test_data()
    hparams = get_test_hparams()

    # Call the train function with sample data and hyperparameters
    model = train(x, y, hparams)

    # Assert logged loss
    assert model.logger.log_dir is not None, "Train loss has not been logged."


def test_model_saved():
    """test that model is saved after training in src.models.train_model.py"""
    x, y = get_normalized_test_data()
    hparams = get_test_hparams()

    # Call the train function with sample data and hyperparameters
    model = train(x, y, hparams)
    save_model(model, get_paths()["model_path"], get_paths()["training_bucket"], push=False)
    # Assert that the model is saved
    assert os.path.exists("./models/model.pth"), "Model not saved."

# the following tests complete test coverage for src.models.train_model.py
