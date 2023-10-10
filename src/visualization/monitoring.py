from http import HTTPStatus
import pandas as pd
import torch

import yaml

# from enum import Enum
from src.models.train_model import load_data
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google.cloud import storage

from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfMissingValues,
    TestAccuracyScore,
    TestPrecisionScore,
    TestRecallScore,
    TestF1Score,
    TestRocAuc,
    TestLogLoss,
    # TestPrecisionByClass,
    # TestRecallByClass,
    # TestF1ByClass,
)
from evidently.report import Report

from src.models.predict_model import load_model

LOCAL = False  # set this to true when developing locally


app = FastAPI()


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


def get_paths():
    with open("./src/configs/config.yaml", "r") as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    return cfg["paths"]


def load_reference_data(from_remote: bool = not LOCAL):
    reference_data = load_data(
        get_paths()["training_data_path"], get_paths()["training_bucket"]
    )
    # sample 1000 rows from reference data
    reference_data = reference_data.sample(1000)
    reference_prediction = reference_data.pop("TAc")
    reference_data.insert(reference_data.shape[1], "TAc", reference_prediction)

    if from_remote:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(get_paths()["inference_bucket"])
        # open the file "database.csv" from the bucket add a new to to the csv, the upload again
        blob = bucket.blob(
            get_paths()["inference_data_path"].split["/"][1]
            + get_paths()["inference_data_path"].split["/"][2]
        )
        blob.download_to_filename(get_paths()["inference_data_path"])
    current_data = pd.read_csv(get_paths()["inference_data_path"])
    current_data.drop(columns=["time"], inplace=True)
    # remove "." and "" from entries in colum input_data of current_data and split to 90 feature columns
    current_data = pd.concat(
        [
            current_data["input_data"]
            .str.strip(".).str.strip(")
            .str.split(",", expand=True),
            current_data["prediction"],
        ],
        axis=1,
    )
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


def load_test_data():
    data = load_data(get_paths()["training_data_path"], get_paths()["training_bucket"])
    # sample 1000 rows from reference data
    data = data.sample(1000)
    data_prediction = data.pop("TAc")
    data.insert(data.shape[1], "target", data_prediction)
    model = load_model()
    input_data = data.drop(columns=["target"]).values
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    # input_tensor = input_tensor.view(1, -1)
    # use model to predict input tensor and add predictions as "prediction" column to data
    with torch.no_grad():
        predictions = model(input_tensor).detach().numpy()
    data.insert(data.shape[1], "prediction", predictions)

    return data


@app.get("/monitoring/", response_class=HTMLResponse)
async def monitoring():
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
        current_data=current_data,
        reference_data=reference_data,
        column_mapping=None,
    )
    data_drift_report.save_html("monitoring.html")

    with open("monitoring.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)


@app.get("/monitoring-tests/", response_class=HTMLResponse)
async def monitoring_tests():
    """Simple get request method that returns a monitoring report."""
    data = load_test_data()

    data_test = TestSuite(
        tests=[
            TestNumberOfMissingValues(),
            TestAccuracyScore(),
            TestPrecisionScore(),
            TestRecallScore(),
            TestF1Score(),
            TestRocAuc(),
            TestLogLoss(),
            #   TestPrecisionByClass(label=0),
            #   TestPrecisionByClass(label=1),
            #   TestRecallByClass(label=0),
            #   TestRecallByClass(label=1),
            #   TestF1ByClass(label=0),
            #   TestF1ByClass(label=1),
        ]
    )
    data_test.run(reference_data=None, current_data=data)

    data_test.save_html("tests.html")

    with open("tests.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)
