import os
import pickle

PROJECT_FOLDER = os.path.abspath(os.path.join(os.getcwd(), "../.."))
ASSET_PATH = os.path.join(PROJECT_FOLDER, "model")


def load_asset(filename, asset_path: str = ASSET_PATH):
    print(f"Loading asset: {filename}")
    with open(os.path.join(asset_path, filename), "rb") as f:
        asset = pickle.load(f)
    return asset
