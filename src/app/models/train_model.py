import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from src.app.models.data import process_data
from model import compute_model_metrics, train_model


def load_data(data_path):
    return pd.read_csv(data_path)


def split_data(data, test_size=0.2):
    train, test = train_test_split(data, test_size=test_size)
    return train, test


def train_and_save_model(
    train, cat_features, label, model_file, encoder_file, binarizer_file
):
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=label, training=True
    )
    model = train_model(X_train, y_train)
    joblib.dump(model, model_file)
    joblib.dump(encoder, encoder_file)
    joblib.dump(lb, binarizer_file)


def evaluate_model(data, cat_features, label, model, encoder, lb, output_dir):
    performance_df = pd.DataFrame(
        columns=["feature", "category", "precision", "recall", "fbeta"]
    )
    for feature in cat_features:
        feature_performance = []
        for category in data[feature].unique():
            subset = data.loc[data[feature] == category]
            X_test, y_test, *_ = process_data(
                subset,
                categorical_features=cat_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb,
            )
            y_pred = model.predict(X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

            feature_performance.append(
                {
                    "feature": feature,
                    "category": category,
                    "precision": precision,
                    "recall": recall,
                    "fbeta": fbeta,
                }
            )
        feature_performance_df = pd.DataFrame(feature_performance)
        performance_df = pd.concat([performance_df, feature_performance_df], axis=0)

    output_file = os.path.join(output_dir, "slice_output.txt")
    performance_df.to_csv(output_file, index=False)


def run():
    # Add code to load in the data.
    data_path = "data/raw/census.csv"
    data = load_data(data_path)

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
    label = "salary"

    train, test = split_data(data)

    # Train and save the model
    model_file = "model.joblib"
    encoder_file = "encoder.joblib"
    binarizer_file = "lb.joblib"
    train_and_save_model(
        train, cat_features, label, model_file, encoder_file, binarizer_file
    )

    # Load the trained model, encoder, and label binarizer
    model = joblib.load(model_file)
    encoder = joblib.load(encoder_file)
    lb = joblib.load(binarizer_file)

    # Evaluate the model
    evaluate_model(test, cat_features, label, model, encoder, lb, output_dir="")
