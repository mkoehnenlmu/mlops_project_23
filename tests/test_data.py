import os
import os.path

import pandas as pd
from typing import List
import pytest
import yaml

from src.models.train_model import load_data, normalize_data
from tests import _PROJECT_ROOT
from tests.define_test_data import get_test_data


def get_cfg() -> dict:
    with open(os.path.join(_PROJECT_ROOT, "src/configs/config.yaml"), "r") as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    return cfg


@pytest.mark.skipif(
    not os.path.exists(get_cfg()["paths"]["training_data_path"]),
    reason="Data files not found",
)
def test_data_shape():
    cfg = get_cfg()
    data = load_data(cfg["paths"]["training_data_path"])
    assert len(data) == 66595, "Dataset did not have the correct number of samples"
    assert (
        data.shape[1] == cfg["hyperparameters"]["input_size"] + 1
    ), "Dataset did not have the correct number of columns"


def test_x_y_split():
    cfg = get_cfg()
    data = get_test_data()
    x, y = normalize_data(data)
    assert x.shape[1] == cfg["hyperparameters"]["input_size"], "Features do not have the correct shape"
    assert len(y.shape) == cfg["hyperparameters"]["output_size"], "Targets do not have the correct shape"


def test_normalization():
    data = get_test_data()
    x, _ = normalize_data(data)

    x_vals: List[list] = []
    for x_i in x:
        x_vals.extend(x_i.tolist())
    assert all(
        (pd.isna(x_val) or (x_val >= 0 and x_val <= 1)) for x_val in x_vals
    ), "Features are not normalized correctly"
