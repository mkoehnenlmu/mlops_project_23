from fastapi.testclient import TestClient
from http import HTTPStatus
from src.models.predict_model import app
import pytest
import json


client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }


@pytest.mark.parametrize(
    "input_data, expected_status_code",
    [
        (("[8000207.0,0,6,6,16,15,9,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
          "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
          "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,650,1]"),
         200),
        (("[8000207.0,0,6,6,16,15,9,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
          "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
          "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,650]"),
         400),  # one feature missing
        (("invalid data"),
         400),  # wrong data form
    ],
)
def test_single_prediction(input_data, expected_status_code):

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
        (["[8000207.0,0,6,6,16,15,9,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
          "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
          "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,650,1]",
          "[8000207.0,0,6,6,16,15,9,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
          "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
          "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,650,1]"],
         200),
        (["[8000207.0,0,6,6,16,15,9,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
          "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
          "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,650]",
          "[8000207.0,0,6,6,16,15,9,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
          "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
          "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,650,1]"],
         400),  # one feature missing
        (["invalid data"],
         400),  # wrong data form
    ],
)
def test_batch_prediction(input_data, expected_status_code):

    # print(len(input_data.split(",")))
    response = client.post(
        "/batch_predict",
        headers={"accept": "application/json",
                 "Content-Type": "application/json"},
        data=json.dumps(input_data)
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
