import os
import os.path

import pandas as pd
from typing import List
import pytest

from src.models.train_model import load_data, normalize_data
from tests.utilities import get_test_data, get_paths, get_hparams


@pytest.mark.skipif(
    not os.path.exists(get_paths()["training_data_path"]),
    reason="Data files not found",
)
def test_data_shape() -> None:
    data = load_data(get_paths()["training_data_path"])
    assert len(data) == 66595, "Dataset did not have the correct number of samples"
    assert (
        data.shape[1] == get_hparams()["input_size"] + 1
    ), "Dataset did not have the correct number of columns"


def test_x_y_split() -> None:
    data = get_test_data()
    x, y = normalize_data(data)
    assert (
        x.shape[1] == get_hparams()["input_size"]
    ), "Features do not have the correct shape"
    assert (
        len(y.shape) == get_hparams()["output_size"]
    ), "Targets do not have the correct shape"


def test_normalization() -> None:
    data = get_test_data()
    x, _ = normalize_data(data)

    x_vals: List[list] = []
    for x_i in x:
        x_vals.extend(x_i.tolist())
    assert all(
        (pd.isna(x_val) or (x_val >= 0 and x_val <= 1)) for x_val in x_vals
    ), "Features are not normalized correctly"
