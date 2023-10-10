import logging
from pathlib import Path
from typing import Any, Dict, Union

import hydra
import numpy as np
from ConfigSpace import Configuration
from smac.facade.hyperparameter_optimization_facade import (
    HyperparameterOptimizationFacade as HPOFacade,
)
from smac.multi_objective.parego import ParEGO
from smac.scenario import Scenario

from src.data.load_data import load_data
from src.models.train_model import evaluate_model, save_model, train
from src.models.tuning.configspace import configspace_new
from src.models.tuning.plot_pareto import plot_pareto

# Global logger
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def evaluate_config(
    config: Configuration, seed: int = 42
) -> Dict[str, Union[str, float]]:
    """
    Evaluate a hyperparameter configuration.

    Args:
        config (Configuration): Configuration to evaluate.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        Dict[str, Union[str, float]]: Dictionary containing loss and size.
    """
    params = dict(config)  # .get_dictionary()

    print("Evaluate config: " + str(params))

    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)

    model = train(train_data, params)

    accuracy, precision, recall, f1, zero_one_loss = evaluate_model(model, test_data)

    print("Loss (F1): " + str(f1.item()))

    # SMAC minimizes, so return 1-F1 score
    return {
        "loss": 1 - f1.item() if not np.isnan(f1.item()) else 1,
        "size": np.log10(sum(p.numel() for p in model.parameters())),
    }


def optimize_configuration(cfg) -> Dict[str, Any]:
    """
    Optimize hyperparameter configuration.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Dict[str, Any]: Dictionary containing the optimized configuration.
    """
    # load the data into a global variable
    global data
    data = load_data(cfg.paths.training_data_path)

    # Scenario object specifying the optimization environment
    scenario = Scenario(
        configspace_new,
        name="HPO2",
        objectives=["loss", "size"],  # multi objective optim
        n_trials=cfg.tuning.num_configs,
        walltime_limit=3600,  # max total time
        n_workers=4,
        output_directory=Path("./models/optimizations/"),
    )

    smac = HPOFacade(
        scenario=scenario,
        target_function=evaluate_config,
        initial_design=HPOFacade.get_initial_design(scenario, n_configs=20),
        multi_objective_algorithm=ParEGO(scenario),
        intensifier=HPOFacade.get_intensifier(scenario, max_config_calls=2),
    )

    incumbents = smac.optimize()

    print("Validated costs from the Pareto front (incumbents):")
    lowest_cost = [
        smac.runhistory.get_configs()[0],
        smac.runhistory.average_cost(smac.runhistory.get_configs()[0]),
    ]
    for incumbent in incumbents:
        cost = smac.validate(incumbent)
        if cost[0] < lowest_cost[1][0]:
            lowest_cost = [incumbent, cost]
        print("---", cost)

    print("Done optimizing. Most promising hyperparam config:")
    print(lowest_cost[0])
    print("With cost: " + str(lowest_cost[1]))

    plot_pareto(smac, incumbents)

    return lowest_cost


def save_config(hparams: Dict[str, Any]) -> None:
    """
    Save hyperparameters to Google Cloud Storage.

    Args:
        hparams (Dict[str, Any]): Hyperparameters to save.

    Returns:
        None
    """
    from google.cloud import storage

    # on Cloud Compute Engine, the service account credentials
    # will be automatically available
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("delay_mlops_data")

    # upload the hyperparameters to the bucket
    blob = bucket.blob("model_configs/hyperparams.json")
    blob.upload_from_string(str(hparams))


def train_optimal_model(cfg, hparams, save=True) -> None:
    """
    Train the optimal model.

    Args:
        cfg: Hydra configuration object.
        hparams: Hyperparameters.
        save (bool, optional): Whether to save the model. Defaults to True.

    Returns:
        None
    """
    # data = load_data(cfg.paths.training_data_path)

    model = train(data, dict(hparams))

    if save:
        save_config(model.hyperparams)
        save_model(model, push=True)


@hydra.main(config_path="../../configs/", config_name="config.yaml", version_base="1.2")
def run_optim(cfg) -> None:
    """
    Main function to run hyperparameter optimization.

    Args:
        cfg: Hydra configuration object.

    Returns:
        None
    """
    optimal_params = optimize_configuration(cfg)
    train_optimal_model(cfg, optimal_params[0], True)


if __name__ == "__main__":
    run_optim()
