import numpy as np
import logging
import hydra

from pathlib import Path

from ConfigSpace import Configuration
from smac.facade.hyperparameter_optimization_facade import (
    HyperparameterOptimizationFacade as HPOFacade,
)
from smac.scenario import Scenario
from smac.multi_objective.parego import ParEGO

# from torch.cuda import is_available as cuda_available

from src.models.tuning.configspace import configspace_new
from src.models.train_model import train, load_data, evaluate_model, save_model
from src.models.tuning.plot_pareto import plot_pareto

# Global logger
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def evaluate_config(config: Configuration, seed: int = 42) -> float:
    params = dict(config)  # .get_dictionary()

    print("Evaluate config: " + str(params))

    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)

    model = train(train_data, params)

    accuracy, precision, recall, f1, zero_one_loss = evaluate_model(model,
                                                                    test_data)

    print("Loss (F1): " + str(f1.item()))

    # SMAC minimizes, so return 1-F1 score
    return {
        "loss": 1 - f1.item() if not np.isnan(f1.item()) else 1,
        "size": np.log10(sum(p.numel() for p in model.parameters())),
    }


def optimize_configuration(cfg):
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


def save_config(hparams: dict):
    from google.cloud import storage

    # on Cloud Compute Engine, the service account credentials
    # will be automatically available
    storage_client = storage.Client()
    bucket = storage_client.get_bucket("delay_mlops_data")

    # upload the hyperparameters to the bucket
    blob = bucket.blob("model_configs/hyperparams.json")
    blob.upload_from_string(str(hparams))


def train_optimal_model(cfg, hparams, save=True):
    # data = load_data(cfg.paths.training_data_path)

    model = train(data, dict(hparams[0]))

    if save:
        save_config(model.hyperparams)
        save_model(model, push=True)


@hydra.main(config_path="../../configs/", config_name="config.yaml",
            version_base="1.2")
def run_optim(cfg):
    optimal_params = optimize_configuration(cfg)
    train_optimal_model(cfg, optimal_params, True)


if __name__ == "__main__":
    run_optim()
