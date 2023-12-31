import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
from ConfigSpace import Configuration
from smac.facade.hyperparameter_optimization_facade import (
    HyperparameterOptimizationFacade as HPOFacade,
)
from smac.multi_objective.parego import ParEGO
from smac.scenario import Scenario

from src.data.load_data import load_data, create_normalized_target
from src.models.train import evaluate_model, save_model, train
from src.models.tuning.configspace import configspace_new
from src.models.tuning.plot_pareto import plot_pareto

from hydra import compose

cfg = compose(config_name="config")

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

    x, y = create_normalized_target(train_data)

    model = train(x, y, params)

    accuracy, precision, recall, f1, zero_one_loss = evaluate_model(model, test_data)

    print("Loss (F1): " + str(f1))

    # SMAC minimizes, so return 1-F1 score
    return {
        "loss": 1 - f1 if not np.isnan(f1) else 1,
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
    data = load_data(cfg.paths.training_data_path, cfg.paths.training_bucket)

    # Scenario object specifying the optimization environment
    scenario = Scenario(
        configspace_new,
        name="HPO2",
        objectives=["loss", "size"],  # multi objective optim
        n_trials=cfg.tuning.num_configs,
        walltime_limit=3600,  # max total time
        n_workers=cfg.tuning.n_workers,  # max parallel workers
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


def save_config(hparams: Dict[str, Any], model_config_path: str) -> None:
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
    with open(model_config_path, "w") as json_file:
        json.dump(hparams, json_file, indent=4)
    blob = bucket.blob(
        model_config_path.split("/")[2] + "/" + model_config_path.split("/")[3]
    )
    blob.upload_from_filename(model_config_path)


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
    global data
    if data is None:
        data = load_data(cfg.paths.training_data_path, cfg.paths.training_bucket)

    x, y = create_normalized_target(data)
    model = train(x, y, dict(hparams))

    if save:
        save_config(model.hyperparams, cfg.paths.model_config_path)
        save_model(model, cfg.paths.model_path, cfg.paths.training_bucket, push=True)


# @hydra.main(config_path="../../configs/", config_name="config.yaml", version_base="1.2")
def run_optim(cfg, save: bool = True) -> None:
    """
    Main function to run hyperparameter optimization.

    Args:
        cfg: Hydra configuration object.

    Returns:
        None
    """
    optimal_params = optimize_configuration(cfg)
    train_optimal_model(cfg, optimal_params[0], save)


def delete_vm() -> None:
    """
    Delete the host virtual machine on Google Cloud Compute Engine.

    Returns:
        None
    """
    import os

    # get current zone
    zone = os.popen("gcloud config get-value compute/zone").read().strip()
    os.system(f"gcloud compute instances delete host --zone={zone} -q")


if __name__ == "__main__":
    run_optim(cfg, save=True)
    delete_vm()
