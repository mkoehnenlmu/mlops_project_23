from fastapi import FastAPI, HTTPException
from google.cloud import storage
import torch
import os
from src.models.model import LightningModel
from typing import List

app = FastAPI()

def download_model_from_gcs():
    # Download the model from Google Cloud Storage
    client = storage.Client()
    bucket = client.get_bucket('delay_mlops_data')
    blob = bucket.blob('models/model.pth')
    blob.download_to_filename('models/model.pth')

async def model_predict(model: LightningModel, input_data: dict):
    # Make the inference
    input_data = input_data.split(',')
    try:
        input_data = [float(x) for x in input_data.values()]
    except:
        raise HTTPException(status_code=400, detail="Invalid input data format")
    input_data = [float(x) for x in input_data]
    input_data = torch.tensor(input_data).float()
    input_data = input_data.unsqueeze(0)
    return(model(input_data))

# write the single_predict function
async def single_predict(input_data: str):
    
    if not os.path.exists('models/model.pth'):
        download_model_from_gcs()

    # Load the model
    model = LightningModel.load_from_checkpoint('models/model.pth')

    # Return the inferred value "delay"
    return {"delay": model_predict(model, input_data)}


@app.get("/predict/{input_data}")
async def predict(input_data: str):
    return single_predict(input_data)

@app.post("/predict")
async def predict(input_data: dict):
    return single_predict(input_data)

@app.post("/batch_predict")
async def batch_predict(input_data: List[dict]):
    
    if not os.path.exists('models/model.pth'):
        download_model_from_gcs()

    model = LightningModel.load_from_checkpoint('models/model.pth')

    # Make the inference
    predictions = []
    for data in input_data:
        prediction = model_predict(model, data)
        predictions.append(prediction[0][0])

    # Return the inferred values "delay"
    return {"delay": predictions}
