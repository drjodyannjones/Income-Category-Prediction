import os
import sys
import pytest
from fastapi.testclient import TestClient
from typing import Any
from pydantic import BaseModel

# Add the path to the src folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from main import app

# Create a TestClient instance to interact with the app
test_client = TestClient(app)


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


# # Define a fixture to provide the TestClient instance to tests
@pytest.fixture()
def client():
    return TestClient(app)


# Define a fixture to provide an example input for incomes below $50K
@pytest.fixture()
def below_50k_example():
    data = CensusData(
        age=39,
        workclass="State-gov",
        fnlgt=77516,
        education="Bachelors",
        education_num=13,
        marital_status="Never-married",
        occupation="Adm-clerical",
        relationship="Not-in-family",
        race="White",
        sex="Male",
        capital_gain=2174,
        capital_loss=0,
        hours_per_week=40,
        native_country="United-States",
    )

    response = test_client.post("/predict", json=data.dict())
    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"


# Define a fixture to provide an example input for incomes above $50K
@pytest.fixture()
def above_50k_example():
    data = CensusData(
        age=50,
        workclass="Self-emp-not-inc",
        fnlgt=83311,
        education="Bachelors",
        education_num=13,
        marital_status="Married-civ-spouse",
        occupation="Exec-managerial",
        relationship="Husband",
        race="White",
        sex="Male",
        capital_gain=0,
        capital_loss=0,
        hours_per_week=13,
        native_country="United-States",
    )

    response = test_client.post("/predict", json=data.dict())
    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"


# Test the root endpoint to ensure it returns a welcome message
def test_root(client: TestClient):
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Income Prediction App!"}


# Test the model endpoint with an example input for incomes below $50K
def test_predict_below_50k(client: TestClient, below_50k_example: dict[str, Any]):
    response = test_client.post("/model", json=below_50k_example)
    assert response.status_code == 200
    assert response.json() == [0]


# Test the model endpoint with an example input for incomes above $50K
def test_predict_above_50k(client: TestClient, above_50k_example: dict[str, Any]):
    response = test_client.post("/model", json=above_50k_example)
    assert response.status_code == 200
    assert response.json() == [1]
