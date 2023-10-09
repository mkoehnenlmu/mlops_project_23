import os
from contextlib import nullcontext as does_not_raise

import pytest
import torch

from src.models.model import LightningModel
from tests.utilities import get_hparams, get_normalized_test_data, get_paths


@pytest.mark.skipif(
    not os.path.exists(get_paths()["training_data_path"]), reason="Data files not found"
)
def test_model_output() -> None:
    x, _ = get_normalized_test_data()

    hparams = get_hparams()

    model = LightningModel(hparams)
    y_pred = model(x[:1])

    assert (
        y_pred.shape[0] == hparams["output_size"]
    ), "Model output has not the correct shape"


@pytest.mark.parametrize(
    "test_input,expectation",
    [
        (torch.randn(1, get_hparams()["input_size"]), does_not_raise()),
        (
            torch.randn(1, 2, 3),
            pytest.raises(ValueError, match="Expected input to a 2D tensor"),
        ),
        (
            torch.randn(20, 30),
            pytest.raises(
                ValueError,
                match=r"Expected each sample to have shape"
                + rf'\[{get_hparams()["input_size"]}\]',
            ),
        ),
    ],
)
def test_error_on_wrong_shape(test_input: torch.Tensor, expectation: any) -> None:
    hparams = get_hparams()
    model = LightningModel(hparams)

    with expectation:
        model(test_input)
