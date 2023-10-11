import csv
import os
from datetime import datetime
from http import HTTPStatus
from typing import Any, Dict, List, Union

import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException
from google.cloud import storage

from src.data.load_data import (
    get_paths,
    get_hparams,
    load_model,
    normalize_data,
    load_data,
    separate_target,
)
from src.models.model import LightningModel

LOCAL = False  # set this to true when developing locally


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


async def process_prediction(
    input_data: List[str],
    model: LightningModel,
    background_tasks: BackgroundTasks,
):
    prediction = await model_predict(model, input_data)
    now = str(datetime.now())
    background_tasks.add_task(
        add_to_database,
        now,
        input_data,
        prediction,
    )
    return prediction


def check_valid_input(input_data: str) -> bool:
    if len(input_data.split(",")) != get_hparams()["input_size"]:
        return False
    else:
        return True


def add_to_database(
    now: str,
    input_data: List[str],
    prediction: float,
    local: bool = LOCAL,
):
    """function that adds the 90 element input and the prediction together with the timestamp now to a csv-file"""
    input_data = input_data.strip('"').strip("'").strip("[").strip("]")
    input_data = input_data.split(",")
    if local:
        if not os.path.exists(get_paths()["inference_data_path"]):
            with open(get_paths()["inference_data_path"], "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["time"]
                    + [f"input{i}" for i in range(len(input_data))]
                    + ["prediction"]
                )
    else:
        # on Cloud Compute Engine, the service account credentials will be automatically available
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(get_paths()["inference_bucket"])
        # open the file "database.csv" from the bucket add a new to to the csv, the upload again
        blob = bucket.blob(
            get_paths()["inference_data_path"].split["/"][1]
            + "/"
            + get_paths()["inference_data_path"].split["/"][2]
        )
        blob.download_to_filename(get_paths()["inference_data_path"])
    with open(get_paths()["inference_data_path"], "a") as f:
        writer = csv.writer(f)
        writer.writerow([now] + input_data + [prediction])
    if not local:
        blob.upload_from_filename(get_paths()["inference_data_path"])


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
        prediction = await process_prediction(input_data, model, background_tasks)

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
    background_tasks: BackgroundTasks,
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
            prediction = await process_prediction(data, model, background_tasks)
            # Return the inferred values "delay"
            predictions.append({"delay": prediction})
        response = {
            "input": input_data,
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "prediction": predictions,
        }

    return response
