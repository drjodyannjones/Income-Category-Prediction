from sklearn.model_selection import train_test_split
from data import process_data
from model import compute_model_metrics, train_model
import pandas as pd
import joblib
import os

# Add code to load in the data.
data_path = "data/raw/census.csv"
data = pd.read_csv(data_path)

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

train, test = train_test_split(data, test_size=0.20)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=label, training=True
)

# Proces the test data with the process_data function.
X_test, y_test, *_ = process_data(
    test,
    categorical_features=cat_features,
    label=label,
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model
model = train_model(X_train, y_train)


def evaluate_model(
    data,
    cat_features: list,
    output_dir: str,
    model=None,
    encoder=None,
    lb=None,
    label=None,
    model_file=None,
    encoder_file=None,
    binarizer_file=None,
):
    model = model or joblib.load(model_file)
    encoder = encoder or joblib.load(encoder_file)
    lb = lb or joblib.load(binarizer_file)

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


# Save the trained model to a file so we can use it in the API
model_file = "model.joblib"
joblib.dump(model, model_file)

# Save the trained encoder to a file so we can use it in the API
encoder_file = "encoder.joblib"
joblib.dump(encoder, encoder_file)

# Save the label binarizer to a file so we can use it in the API
binarizer_file = "lb.joblib"
joblib.dump(lb, binarizer_file)

# Evaluate the model
evaluate_model(
    test,
    cat_features,
    output_dir="",
    label=label,
    model_file=model_file,
    encoder_file=encoder_file,
    binarizer_file=binarizer_file,
)
