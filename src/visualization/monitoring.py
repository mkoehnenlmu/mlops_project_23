from http import HTTPStatus

import pandas as pd
import torch
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestAccuracyScore,
    TestF1ByClass,
    TestF1Score,
    TestLogLoss,
    TestNumberOfMissingValues,
    TestPrecisionByClass,
    TestPrecisionScore,
    TestRecallByClass,
    TestRecallScore,
    TestRocAuc,
)
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google.cloud import storage

# from enum import Enum
from src.data.load_data import get_additional_configs, get_paths, load_data
from src.models.model import LightningModel
from src.models.predict_model import load_model

LOCAL = True  # set this to true when developing locally


app = FastAPI()


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


def load_reference_data(from_remote: bool = not LOCAL):
    reference_data = load_data(n_rows=10000)
    # sample 1000 rows from reference data
    reference_data = reference_data.sample(100)
    reference_prediction = reference_data.pop(get_additional_configs()["dependent_var"])
    reference_data.insert(
        reference_data.shape[1],
        get_additional_configs()["dependent_var"],
        reference_prediction,
    )

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

    current_data = pd.concat(
        [
            current_data["input_data"]
            .str.strip("[")
            .str.strip("]")
            .str.split(",", expand=True),
            current_data["prediction"],
        ],
        axis=1,
    )
    current_data.columns = reference_data.columns
    # TODO: dtypes are "object" which is not convertible to int
    # convert type of columns of current_data to the same type as reference_data if not nan
    for col in current_data.columns:
        try:
            # current_data[col] = current_data[col].astype(str)
            # pd.to_numeric(current_data[col], errors='raise', downcast=reference_data[col].dtype)
            current_data[col] = current_data[col].astype(reference_data[col].dtype)
        except Exception:
            print(f"problem with dtype of col {col}")

    print(current_data.dtypes)

    return reference_data, current_data


load_reference_data()


def predict_row(row: pd.Series, model: LightningModel):
    input_tensor = torch.tensor(row.values, dtype=torch.float32)
    input_tensor = input_tensor.view(1, -1)
    with torch.no_grad():
        prediction = model.forward(input_tensor)
    # if prediction.item() >= 0.5:
    #    return 1
    # else:
    #    return 0
    return prediction.item()


def load_test_data():
    data = load_data(n_rows=10000)
    # sample 1000 rows from reference data
    data = data.sample(1000)
    data_prediction = data.pop(get_additional_configs()["dependent_var"])
    data.insert(data.shape[1], "target", data_prediction)
    model = load_model()
    input_data = data.drop(columns=["target"])
    data["prediction"] = input_data.apply(lambda row: predict_row(row, model), axis=1)

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
            TestPrecisionByClass(label=0),
            TestPrecisionByClass(label=1),
            TestRecallByClass(label=0),
            TestRecallByClass(label=1),
            TestF1ByClass(label=0),
            TestF1ByClass(label=1),
        ]
    )
    data_test.run(reference_data=None, current_data=data)

    data_test.save_html("tests.html")

    with open("tests.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content, status_code=200)
