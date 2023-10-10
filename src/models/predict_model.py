from http import HTTPStatus
from typing import Any, Dict, List, Union

import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException

from src.data.load_data import get_hparams, load_model, normalize_data, load_data, separate_target
from src.models.model import LightningModel

app = FastAPI()


@app.get("/")
def root() -> Dict[str, Union[str, HTTPStatus]]:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


async def model_predict(model: LightningModel, input_data: str) -> torch.Tensor:
    # Make the inference
    input_data = input_data.strip('"').strip("'").strip("[").strip("]")
    input_data = input_data.split(",")
    try:
        input_data = [float(x) for x in input_data]
    except HTTPException:
        raise HTTPException(status_code=400, detail="Invalid input data format")

    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    input_tensor = input_tensor.view(1, -1)

    # Normalize the input data with the existing training data
    train_data = load_data()
    x, y = separate_target(train_data)
    # add input tensor at the bottom of x
    x = torch.cat((x, input_tensor), 0)
    norm_x = normalize_data(x)

    # get the last row of the normalized data
    norm_tensor = norm_x[-1, :]

    input = norm_tensor.view(1, -1)
    prediction = model.forward(input)

    return prediction.item()


def check_valid_input(input_data: str) -> bool:
    if len(input_data.split(",")) != get_hparams()["input_size"]:
        return False
    else:
        return True


@app.post("/predict")
async def predict(
    input_data: str,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    if not check_valid_input(input_data):
        response = {
            "input": input_data,
            "message": "The provided input data does not match the required format",
            "status-code": HTTPStatus.BAD_REQUEST,
            "prediction": None,
        }
    else:
        model = load_model()
        prediction = await model_predict(model, input_data)
        # Return the inferred values "delay"
        response = {
            "input": input_data,
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "prediction": [{"delay": prediction}],
        }

    return response


@app.post("/batch_predict")
async def batch_predict(
    input_data: List[str],
) -> Dict[str, Any]:
    model = load_model()
    if not all(check_valid_input(data) for data in input_data):
        response = {
            "input": input_data,
            "message": "The provided input data does not match the required format",
            "status-code": HTTPStatus.BAD_REQUEST,
            "prediction": None,
        }
    else:
        # Make the inference
        predictions: List[Dict[str, torch.Tensor]] = []
        for data in input_data:
            prediction = await model_predict(model, data)
            # Return the inferred values "delay"
            predictions.append({"delay": prediction})
        response = {
            "input": input_data,
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "prediction": predictions,
        }

    return response
