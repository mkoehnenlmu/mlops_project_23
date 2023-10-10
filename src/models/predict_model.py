import csv
import os
from http import HTTPStatus
from typing import Any, Dict, List, Union

import torch
from datetime import datetime
import pandas as pd

from src.data.load_data import load_data, get_paths
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report

from src.data.load_data import (
    get_hparams,
    load_model,
    normalize_data,
    load_data,
    separate_target,
)
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


async def process_prediction(input_data: List[str],
                             model: LightningModel,
                             background_tasks: BackgroundTasks,) -> float:

    prediction = await model_predict(model, input_data)
    now = str(datetime.now())
    background_tasks.add_task(
        add_to_database,
        now,
        input_data,
        prediction.item(),
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
    local: bool = False,
) -> None:
    """function that adds the 90 element input and the prediction together with the timestamp now to a csv-file"""
    if local:
        if not os.path.exists("database.csv"):
            with open("database.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "input_data", "prediction"])
    else:
        # on Cloud Compute Engine, the service account credentials will be automatically available
        storage_client = storage.Client()
        bucket = storage_client.get_bucket("delay_mlops_inference")
        # open the file "database.csv" from the bucket add a new to to the csv, the upload again
        blob = bucket.blob("database.csv")
        blob.download_to_filename("database.csv")
    with open("database.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([now, input_data, prediction])
    if not local:
        blob.upload_from_filename("database.csv")


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


def load_reference_data():
    reference_data = load_data(get_paths()["training_data_path"])
    # sample 1000 rows from reference data
    reference_data = reference_data.sample(1000)
    reference_prediction = reference_data.pop('TAc')
    reference_data.insert(reference_data.shape[1], "TAc", reference_prediction)

    if from_remote:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket("delay_mlops_inference")
        # open the file "database.csv" from the bucket add a new to to the csv, the upload again
        blob = bucket.blob("database.csv")
        blob.download_to_filename("database.csv")
    current_data = pd.read_csv('database.csv')
    current_data.drop(columns=['time'], inplace=True)
    # remove "." and "" from entries in colum input_data of current_data and split to 90 feature columns
    current_data = pd.concat([current_data['input_data'].str.strip(".).str.strip(").str.split(',', expand=True),
                              current_data['prediction']], axis=1)
    current_data.columns = reference_data.columns
    # convert type of columns of current_data to the same type as reference_data if not nan

    for col in current_data.columns:
        try:
            # pd.to_numeric(current_data[col], errors='raise', downcast=reference_data[col].dtype)
            # we currently cannot deal with nans here!
            current_data[col] = current_data[col].astype(reference_data[col].dtype)
        except ValueError:
            print(col)

    return reference_data, current_data


@app.get("/monitoring/", response_class=HTMLResponse)
async def monitoring() -> HTMLResponse:
    """Simple get request method that returns a monitoring report."""
    reference_data, current_data = load_reference_data()

    data_drift_report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            TargetDriftPreset(),
        ]
    )

    data_drift_report.run(
        current_data=reference_data,
        reference_data=current_data,
        column_mapping=None,
    )
    data_drift_report.save_html("monitoring.html")

    with open("monitoring.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)
