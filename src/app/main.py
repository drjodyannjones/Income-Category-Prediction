import pandas as pd
from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib

from app.models.data import process_data
from app.models.model import inference
from app.models.train_model import train_model
from app.config import Settings


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


def load_assets(settings: Settings):
    model = joblib.load(settings.model_path)
    encoder = joblib.load(settings.encoder_path)
    lb = joblib.load(settings.lb_path)
    return model, encoder, lb


@app.get("/")
async def root():
    return JSONResponse(
        status_code=200,
        content={"message": "Welcome to Income Prediction App!"},
    )


@app.post("/model/")
async def predict(data: InputData, assets=Depends(load_assets)):
    model, encoder, lb = assets

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
    return JSONResponse(status_code=200, content=predictions.tolist())


if __name__ == "__main__":
    import uvicorn

    settings = Settings()
    model, encoder, lb = load_assets(settings)

    app.state.settings = settings
    app.state.model = model
    app.state.encoder = encoder
    app.state.lb = lb

    uvicorn.run(app, host="0.0.0.0", port=8000)
