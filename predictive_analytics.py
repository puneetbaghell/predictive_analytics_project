from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "historical_sales.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def load_and_clean_data(path):
    df = pd.read_csv(path)
    required = {"Date", "Sales"}
    if not required.issubset(df.columns):
        raise ValueError("Dataset must contain exactly the required columns: Date and Sales.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Date", "Sales"])
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date")
    df = df[df["Sales"] >= 0].copy()
    if len(df) < 12:
        raise ValueError("Please provide at least 12 valid monthly observations.")
    return df.reset_index(drop=True)

def create_features(df):
    data = df.copy()
    data["Time_Index"] = np.arange(len(data))
    data["Month"] = data["Date"].dt.month
    data["Month_Sin"] = np.sin(2 * np.pi * data["Month"] / 12)
    data["Month_Cos"] = np.cos(2 * np.pi * data["Month"] / 12)
    return data

def main():
    df = load_and_clean_data(DATA_PATH)
    featured = create_features(df)

    # Keep the latest 20% for testing; never shuffle time-series observations.
    split_at = max(1, int(len(featured) * 0.8))
    train = featured.iloc[:split_at].copy()
    test = featured.iloc[split_at:].copy()
    if len(test) == 0:
        raise ValueError("Not enough observations for a test set.")

    feature_cols = ["Time_Index", "Month_Sin", "Month_Cos"]
    model = LinearRegression()
    model.fit(train[feature_cols], train["Sales"])
    test_predictions = model.predict(test[feature_cols])

    mae = mean_absolute_error(test["Sales"], test_predictions)
    rmse = np.sqrt(mean_squared_error(test["Sales"], test_predictions))
    r2 = r2_score(test["Sales"], test_predictions) if len(test) >= 2 else float("nan")

    metrics = pd.DataFrame([{
        "MAE": mae, "RMSE": rmse, "R2": r2,
        "Training Rows": len(train), "Testing Rows": len(test)
    }])
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    test_results = test[["Date", "Sales"]].copy()
    test_results["Predicted_Sales"] = test_predictions
    test_results.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    # Forecast the next 12 months.
    future_dates = pd.date_range(df["Date"].max() + pd.offsets.MonthBegin(1), periods=12, freq="MS")
    future = pd.DataFrame({"Date": future_dates})
    future["Time_Index"] = np.arange(len(df), len(df) + len(future))
    future["Month"] = future["Date"].dt.month
    future["Month_Sin"] = np.sin(2 * np.pi * future["Month"] / 12)
    future["Month_Cos"] = np.cos(2 * np.pi * future["Month"] / 12)

    # Retrain on all available history for the final forecast.
    final_model = LinearRegression()
    final_model.fit(featured[feature_cols], featured["Sales"])
    future["Forecast_Sales"] = np.maximum(0, final_model.predict(future[feature_cols]))
    future[["Date", "Forecast_Sales"]].to_csv(OUTPUT_DIR / "future_forecast.csv", index=False)

    # Save a clean actual-vs-predicted chart.
    plt.figure(figsize=(11, 5))
    plt.plot(train["Date"], train["Sales"], label="Training historical sales")
    plt.plot(test["Date"], test["Sales"], label="Actual test sales", marker="o")
    plt.plot(test["Date"], test_predictions, label="Predicted test sales", marker=".")
    plt.title("Historical Sales: Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # Save the future forecast chart.
    plt.figure(figsize=(11, 5))
    plt.plot(df["Date"], df["Sales"], label="Historical sales")
    plt.plot(future["Date"], future["Forecast_Sales"], label="12-month forecast", marker="o")
    plt.title("12-Month Sales Forecast")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "future_forecast.png", dpi=150)
    plt.close()

    print("\\nPREDICTIVE ANALYTICS USING HISTORICAL DATA")
    print("-" * 48)
    print(f"Valid historical rows: {len(df)}")
    print(f"Training rows: {len(train)} | Testing rows: {len(test)}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.3f}")
    print("\\nNext 12 months forecast:")
    print(future[["Date", "Forecast_Sales"]].to_string(index=False))
    print(f"\\nFiles saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
