# tests/test_train.py
import pytest
from sklearn.linear_model import LinearRegression
from joblib import load
import os

def test_model_exists():
    assert os.path.exists("model.joblib"), "Trained model not found."

def test_model_type():
    model = load("model.joblib")
    assert isinstance(model, LinearRegression), "Model is not LinearRegression."

def test_model_has_coef():
    model = load("model.joblib")
    assert hasattr(model, 'coef_'), "Model has no coefficients."
