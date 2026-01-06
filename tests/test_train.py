import os
import sys
import joblib

# Add project root to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.train import train_model


def test_model_training():
    train_model()
    assert os.path.exists("models/model.pkl")

    model = joblib.load("models/model.pkl")
    assert hasattr(model, "predict")
