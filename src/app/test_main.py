import os
import sys
import pytest
from fastapi.testclient import TestClient
from src.app.main import app
from typing import Any

# Add the path to the src folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Create a TestClient instance to interact with the app
client = TestClient(app)


# Define a fixture to provide the TestClient instance to tests
@pytest.fixture()
def client():
    return TestClient(app)


# Define a fixture to provide an example input for incomes below $50K
@pytest.fixture()
def below_50k_example():
    return {
        "age": 45,
        "fnlgt": 2334,
        "education-num": 13,
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 60,
        "workclass": "State-gov",
        "education": "Bachelors",
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Wife",
        "race": "Black",
        "sex": "Female",
        "native-country": "Cuba",
    }


# Define a fixture to provide an example input for incomes above $50K
@pytest.fixture()
def above_50k_example():
    return {
        "age": 52,
        "fnlgt": 209642,
        "education-num": 9,
        "capital-gain": 123387,
        "capital-loss": 0,
        "hours-per-week": 40,
        "workclass": "Self-emp-not-inc",
        "education": "Bachelors",
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "native-country": "United-States",
    }


# Test the root endpoint to ensure it returns a welcome message
def test_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Income Prediction App!"}


# Test the model endpoint with an example input for incomes below $50K
def test_predict_below_50k(client: TestClient, below_50k_example: dict[str, Any]):
    response = client.post("/model/", json=below_50k_example)
    assert response.status_code == 200
    assert response.json() == [0]


# Test the model endpoint with an example input for incomes above $50K
def test_predict_above_50k(client: TestClient, above_50k_example: dict[str, Any]):
    response = client.post("/model/", json=above_50k_example)
    assert response.status_code == 200
    assert response.json() == [1]
