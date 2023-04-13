# This script trains the machine learning model and saves it to a joblib file.

from sklearn.model_selection import train_test_split
from data import process_data
import pandas as pd
from model import train_model
import joblib

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv(
    "/Users/drjodyannjones/Documents/Projects/IncomeCategoryPrediction/data/raw/census.csv"
)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model
model = train_model(X_train, y_train)

# Save the trained model to a file so we can use it in the API
joblib.dump(model, "model.joblib")

# Save the trained encoder to a file so we can use it in the API
joblib.dump(encoder, "encoder.joblib")
