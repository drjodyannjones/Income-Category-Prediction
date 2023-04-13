import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic import Field
import joblib
from .models.data import process_data
from .models.model import inference
from .models.train_model import train_model
from .config import Settings
from .utils.utils import load_asset


app = FastAPI()


class InputData(BaseModel):
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


@app.post("/model/")
async def predict(data: InputData):
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

    # Modify to pass in multiple indices
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
    return JSONResponse(status_code=200, content=predictions.tolist())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
