from fastapi import FastAPI, HTTPException
from http import HTTPStatus
from google.cloud import storage

import torch
import os
from src.models.model import LightningModel
from typing import List

# from enum import Enum
import yaml


app = FastAPI()


@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


def download_model_from_gcs():
    # Download the model from Google Cloud Storage
    client = storage.Client()
    bucket = client.get_bucket('delay_mlops_data')
    blob = bucket.blob('models/model.pth')
    blob.download_to_filename('models/model.pth')


def get_hparams():
    with open("./src/configs/config.yaml", "r") as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    hparams = {"lr": cfg["hyperparameters"]["learning_rate"],
               "epochs": cfg["hyperparameters"]["epochs"],
               "batch_size": cfg["hyperparameters"]["batch_size"],
               "input_size": cfg["hyperparameters"]["input_size"],
               "output_size": cfg["hyperparameters"]["output_size"],
               "hidden_size": cfg["hyperparameters"]["hidden_size"],
               "num_layers":  cfg["hyperparameters"]["num_layers"],
               "criterion":  cfg["hyperparameters"]["criterion"],
               "optimizer":  cfg["hyperparameters"]["optimizer"],
               }
    return hparams


def load_model():
    if not os.path.exists('./models/model.pth'):
        download_model_from_gcs()
    hparams = get_hparams()
    # Load the model
    model = LightningModel(hparams=hparams)
    loaded_state_dict = torch.load('./models/model.pth')
    model.load_state_dict(loaded_state_dict)

    return model


async def model_predict(model: LightningModel, input_data: str):
    # Make the inference
    input_data = input_data.strip('"').strip("'").strip("[").strip("]")
    input_data = input_data.split(',')
    try:
        input_data = [float(x) for x in input_data]
    except HTTPException:
        raise HTTPException(status_code=400,
                            detail="Invalid input data format")

    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    input_tensor = input_tensor.view(1, -1)
    prediction = model.forward(input_tensor)

    return prediction


def check_valid_input(input_data: str):
    if len(input_data.split(",")) != 90:
        return False
    else:
        return True


@app.post("/predict")
async def predict(input_data: str):
    if not check_valid_input(input_data):
        response = {
            "input": input_data,
            "message": "The provided input data does not match"
            + "the required format",
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
async def batch_predict(input_data: List[str]):

    model = load_model()
    if not all(check_valid_input(data) for data in input_data):
        response = {
            "input": input_data,
            "message": "The provided input data does not match"
            + "the required format",
            "status-code": HTTPStatus.BAD_REQUEST,
            "prediction": None,
        }
    else:
        # Make the inference
        predictions = []
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
