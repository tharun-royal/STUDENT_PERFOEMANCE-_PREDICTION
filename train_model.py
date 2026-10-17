from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_performance.csv"
MODEL_PATH = ROOT / "student_score_model.joblib"

def main():
    data = pd.read_csv(DATA_PATH)
    features = [
        "hours_studied",
        "attendance_percent",
        "previous_score",
        "assignments_completed",
    ]
    target = "final_score"

    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    joblib.dump({"model": model, "features": features}, MODEL_PATH)
    print("Model training complete.")
    print(f"Training rows: {len(X_train)} | Testing rows: {len(X_test)}")
    print(f"MAE:  {mae:.2f} points")
    print(f"RMSE: {rmse:.2f} points")
    print(f"R²:   {r2:.3f}")
    print(f"Saved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
