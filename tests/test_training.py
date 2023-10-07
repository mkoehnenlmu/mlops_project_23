import torch

from src.models.model import LightningModel
from src.models.train import train
from tests.utilities import get_hparams, get_normalized_test_data


def test_model_created() -> None:
    x, y = get_normalized_test_data()
    hparams = get_hparams()
    # Call the train function with sample data and hyperparameters
    model = train(x, y, hparams)

    # Assert that the returned model is not None
    assert model is not None, "Model not instantiated."
    assert isinstance(model, LightningModel), "Model not instance of specified model."


def test_model_hyperparameters() -> None:
    x, y = get_normalized_test_data()
    hparams = get_hparams()
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
    hparams = get_hparams()

    # Call the train function with sample data and hyperparameters
    model = train(x, y, hparams)

    # Assert logged loss
    assert model.logger.log_dir is not None, "Train loss has not been logged."


@pytest.mark.skipif(not os.path.exists(get_paths()["training_data_path"]),
                    reason="Data files not found")
def test_model_saved():
    """test that model is saved after training in src.models.train_model.py"""
    data = load_data(get_paths()["training_data_path"])
    sample_data = data[:5]
    hparams = get_hparams()

    # Call the train function with sample data and hyperparameters
    train(sample_data, hparams)

    # Assert that the model is saved
    assert os.path.exists("./models/model.pth"), "Model not saved."

# the following tests complete test coverage for src.models.train_model.py
