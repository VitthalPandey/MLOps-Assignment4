# src/utils.py

from joblib import dump, load

def save_model(model, filename="model.joblib"):
    """Saves model to a joblib file."""
    dump(model, filename)

def load_model(filename="model.joblib"):
    """Loads model from a joblib file."""
    return load(filename)

def print_metrics(y_true, y_pred):
    """Prints R² score and MSE."""
    from sklearn.metrics import mean_squared_error, r2_score
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"R² Score: {r2:.4f}, MSE: {mse:.4f}")
    return r2, mse
