import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use relative imports
from app.utils.utils import load_asset
from app.config import Settings
from app.models.train_model import train_model
from app.models.model import inference
from app.models.data import process_data

import joblib
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import FastAPI
import pandas as pd
import asyncio

app = FastAPI()


class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


class PredictionResult(BaseModel):
    prediction: str


@app.on_event("startup")
async def startup_event():
    app.state.settings = Settings()
    app.state.model = joblib.load(app.state.settings.model_path)
    app.state.encoder = joblib.load(app.state.settings.encoder_path)
    app.state.lb = joblib.load(app.state.settings.lb_path)


@app.get("/")
async def root():
    return JSONResponse(
        status_code=200,
        content={"message": "Welcome to Income Prediction App!"},
    )


@app.post("/model", response_model=PredictionResult)
async def predict(data: CensusData):
    model = app.state.model
    encoder = app.state.encoder
    lb = app.state.lb

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    df = pd.DataFrame(data.dict(by_alias=True), index=[0])
    X, *_ = process_data(
        df,
        categorical_features=cat_features,
        label=None,
        encoder=encoder,
        lb=lb,
        training=False,
    )
    predictions = inference(model, X)
    prediction = lb.inverse_transform(predictions)[0]
    return PredictionResult(prediction=prediction)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
