import joblib
import numpy as np

def evaluate_model():
    model = joblib.load("models/model.pkl")
    test_x = np.array([[6], [7], [8]])  # Ex test data
    predictions = model.predict(test_x)

    print("Predictions: ", predictions)

if __name__ == "__main__":
    evaluate_model()