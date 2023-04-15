import json
import requests

url = "https://income-category-prediction.onrender.com/model"

sample = {
    "workclass": "Private",
    "education": "Bachelors",
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Wife",
    "race": "Black",
    "sex": "Female",
    "native_country": "Cuba",
    "age": 23,
    "fnlgt": 2334,
    "education_num": 7,
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 60,
}

headers = {"Content-type": "application/json"}
response = requests.post(url, data=json.dumps(sample), headers=headers)
print("Status Code: ", response.status_code)
print("Prediction: ", response.text)
