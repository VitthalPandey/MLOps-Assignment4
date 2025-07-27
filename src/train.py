# src/train.py
import src.utils as utils
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import src.utils as utils  # ✅ import your reusable helpers

def main():
    # Load dataset
    X, y = fetch_california_housing(return_X_y=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    r2, mse = utils.print_metrics(y_test, y_pred)

    # Save the model
    utils.save_model(model, filename="model.joblib")

    print("Model training complete and saved as model.joblib")

if __name__ == "__main__":
    main()

