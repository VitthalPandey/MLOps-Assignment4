# src/predict.py

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from joblib import load

def dequantize_array(q_arr, min_val, max_val):
    scale = (max_val - min_val) / 255
    return (q_arr.astype(np.float32) * scale) + min_val

# Load dataset
X, y = fetch_california_housing(return_X_y=True)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load quantized parameters
params = load("quant_params.joblib")
coef = dequantize_array(params["q_coef"], params["coef_min"], params["coef_max"])
intercept = dequantize_array(np.array([params["q_intercept"]]), params["intercept_min"], params["intercept_max"])[0]

# Predict
y_pred = np.dot(X_test, coef) + intercept
print("Sample predictions (first 5):", y_pred[:5])
