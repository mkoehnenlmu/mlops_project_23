import json
from http import HTTPStatus

import pytest
from fastapi.testclient import TestClient

import src.models.predict_model
from src.models.model import LightningModel
from src.models.predict_model import app
from tests.utilities import get_hparams, get_inference_test_data, get_test_data

client = TestClient(app)


# define that a mock model is used when calling in predict or batch_predict on the client
def mock_load_model():
    return LightningModel(get_hparams())


# define that a empty file is returned when calling load_data in predict or batch_predict on the client
def mock_load_data():
    return get_test_data()


def test_read_main():
    response = client.get("/")
    assert response.json() == {
        "message": HTTPStatus.OK.phrase,
        "status-code": "200",
    }


@pytest.mark.parametrize(
    "input_data, expected_status_code",
    [
        (
            (f"{get_inference_test_data()}"),
            200,
        ),
        (
            (f"{get_inference_test_data()[:-1]}"),
            400,
        ),  # one feature missing
        (("invalid data"), 400),  # wrong data form
    ],
)
def test_single_prediction(input_data, expected_status_code, monkeypatch):
    monkeypatch.setattr(src.models.predict_model, "load_data", mock_load_data)
    monkeypatch.setattr(src.models.predict_model, "load_model", mock_load_model)

    print(len(input_data.split(",")))
    response = client.post(
        "/predict",
        params={"input_data": input_data},
        headers={"accept": "application/json"},
    )
    print(response.json())
    print(expected_status_code)
    print(response.status_code)
    print(response.json()["status-code"])
    assert response.json()["status-code"] == expected_status_code

    if expected_status_code == 200:
        assert "prediction" in response.json()
        assert "delay" in response.json()["prediction"][0]
    elif expected_status_code == 400:
        assert "message" in response.json()
        assert "input data does not match" in response.json()["message"]


@pytest.mark.parametrize(
    "input_data, expected_status_code",
    [
        (
            [f"{get_inference_test_data()}", f"{get_inference_test_data()}"],
            200,
        ),
        (
            [f"{get_inference_test_data()[:-1]}", f"{get_inference_test_data()}"],
            400,
        ),  # one feature missing
        (["invalid data"], 400),  # wrong data form
    ],
)
def test_batch_prediction(input_data, expected_status_code, monkeypatch):
    monkeypatch.setattr(src.models.predict_model, "load_data", mock_load_data)
    monkeypatch.setattr(src.models.predict_model, "load_model", mock_load_model)

    # print(len(input_data.split(",")))
    response = client.post(
        "/batch_predict",
        headers={"accept": "application/json", "Content-Type": "application/json"},
        data=json.dumps(input_data),
    )
    print(response.json())
    print(expected_status_code)
    print(response.status_code)
    print(response.json()["status-code"])
    assert response.json()["status-code"] == expected_status_code

    if expected_status_code == 200:
        assert "prediction" in response.json()
        assert "delay" in response.json()["prediction"][0]
    elif expected_status_code == 400:
        assert "message" in response.json()
        assert "input data does not match" in response.json()["message"]
