import os
import joblib
from src.train import train_model

def test_model_training():
    # Ensure the models directory is clean before the test runs
    train_model()
    assert os.path.exists("models/model.pkl")

    model = joblib.load("models/model.pkl")
    assert hasattr(model, "predict") # Check if the model has a predict method 
