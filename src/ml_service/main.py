from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from utils import feature_engineer, encode
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained model (replace 'model/model.pkl' with your model's actual path)
with open('model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('model/meta.pkl', 'rb') as model_file:
    metadict = pickle.load(model_file)
    train_metrics = metadict["train_metrics"]
    ohe = metadict["ohe"]
    mms = metadict["mms"]

# Define the request body schema using Pydantic
class PredictRequest(BaseModel):
    street: str
    project: str
    marketsegment: str
    x: str
    y: str
    area: str
    floorrange: str
    noofunits: str
    contractdate: str
    typeofsale: str
    district: str
    typeofarea: str
    tenure: str

# Define the API endpoint for predictions
@app.post("/predict/")
async def predict(request: PredictRequest):
    # try:

    # Convert input data into a NumPy array for the model
    input_data = pd.DataFrame([[
        request.street, 
        request.project, 
        request.marketsegment, 
        request.x, 
        request.y,
        request.area, 
        request.floorrange,
        request.noofunits, 
        request.contractdate,
        request.typeofsale, 
        request.district, 
        request.typeofarea,
        request.tenure,
    ]], columns=[request.schema()['required']])

    # filter and transform
    dataset = feature_engineer(input_data, predict=True)
    dataset_encoded = encode(dataset, ohe, mms, fit=False)

    # Get the prediction from the model
    prediction = model.predict(dataset_encoded)
            
    # Return the prediction as a response
    return {"prediction": prediction.tolist()}
    
    # except Exception as e:
    #     # Handle errors
    #     raise HTTPException(status_code=500, detail=str(e))


# Basic health check endpoint
@app.get("/")
async def health_check():
    return {"status": "ok"}
