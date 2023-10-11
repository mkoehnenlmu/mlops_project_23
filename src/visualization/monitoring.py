from http import HTTPStatus
import os
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

# from enum import Enum
from src.data.load_data import get_additional_configs, get_paths, load_data, download_file_from_gcs
from src.models.model import LightningModel
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


def load_reference_data(column_set: tuple = None, sample_size: int = 1000,):
    reference_data = load_data(n_rows=10000)
    # sample 1000 rows from reference data
    reference_data = reference_data.sample(sample_size)
    reference_prediction = reference_data.pop(get_additional_configs()["dependent_var"])
    reference_data.insert(
        reference_data.shape[1],
        get_additional_configs()["dependent_var"],
        reference_prediction,
    )

    if not os.path.exists(get_paths()["inference_data_path"]):
        download_file_from_gcs(get_paths()["inference_data_path"].split("/")[1]
                               + "/"
                               + get_paths()["inference_data_path"].split("/")[2],
                               get_paths()["inference_data_path"],
                               get_paths()["inference_bucket"])
    current_data = pd.read_csv(get_paths()["inference_data_path"])

    current_data.drop(columns=["time"], inplace=True)
    current_data.columns = reference_data.columns

    # convert type of columns of current_data to the same type as reference_data if not nan
    for col in current_data.columns:
        try:
            current_data[col] = current_data[col].astype(reference_data[col].dtype)
        except Exception:
            print(f"problem with dtype of col {col}")

    if column_set:
        return reference_data.iloc[:, column_set[0]:column_set[1]], current_data.iloc[:, column_set[0]:column_set[1]]
    else:
        return reference_data, current_data


def predict_row(row: pd.Series, model: LightningModel):
    input_tensor = torch.tensor(row.values, dtype=torch.float32)
    input_tensor = input_tensor.view(1, -1)
    with torch.no_grad():
        prediction = model.forward(input_tensor)

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
    reference_data, current_data = load_reference_data(column_set=(0, 10))
    # get only first 10 columns of reference and current
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
