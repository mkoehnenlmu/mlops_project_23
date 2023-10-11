from typing import Any, Dict

from torch import backends

from src.data.load_data import load_data, create_normalized_target
from src.models.train import evaluate_model, save_model, train

from hydra import compose

cfg = compose(config_name="config")


# @hydra.main(config_path="../configs/", config_name="config.yaml", version_base="1.2")
def main(cfg: Dict[str, Any]) -> None:
    """
    Main function to train a model using Hydra configuration.

    Args:
        cfg (Dict[str, Any]): Hydra configuration object.

    Returns:
        None
    """
    # get data
    data = load_data(cfg.paths.training_data_path)
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)

    x, y = create_normalized_target(train_data, "DEP_DEL15")
    x_test, y_test = create_normalized_target(test_data, "DEP_DEL15")

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

    model = train(x, y, x_test, y_test, hparams)

    acc, pred, recall, f1, zol = evaluate_model(model, test_data)

    print(
        "Model training loss (Accuracy, Precision, Recall, F1, 0-1 Loss): "
        + str(acc)
        + ", "
        + str(pred)
        + ", "
        + str(recall)
        + ", "
        + str(f1)
        + ", "
        + str(zol)
    )

    # TODO save scores with hyperparams in database

    save_model(model, cfg.paths.model_path, cfg.paths.training_bucket, push=True)


if __name__ == "__main__":
    main(cfg)
