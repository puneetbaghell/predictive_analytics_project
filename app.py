from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "outputs"
DATA = BASE_DIR / "data" / "historical_sales.csv"

st.set_page_config(page_title="Predictive Analytics Dashboard", page_icon="📈", layout="wide")
st.title("📈 Predictive Analytics Using Historical Data")
st.write("Explore historical sales, evaluate a forecasting model, and review the next 12 months' forecast.")

if not (OUT / "future_forecast.csv").exists():
    st.warning("Forecast files are missing. Run `python predictive_analytics.py` first.")
    st.stop()

history = pd.read_csv(DATA, parse_dates=["Date"])
forecast = pd.read_csv(OUT / "future_forecast.csv", parse_dates=["Date"])
metrics = pd.read_csv(OUT / "model_metrics.csv")
test = pd.read_csv(OUT / "test_predictions.csv", parse_dates=["Date"])

a, b, c = st.columns(3)
a.metric("Historical records", f"{len(history):,}")
b.metric("Mean absolute error (MAE)", f"{metrics.loc[0, 'MAE']:.2f}")
c.metric("RMSE", f"{metrics.loc[0, 'RMSE']:.2f}")

st.subheader("Historical Sales")
st.line_chart(history.set_index("Date")["Sales"])

st.subheader("Actual vs Predicted (Test Period)")
plot_test = test.set_index("Date")[["Sales", "Predicted_Sales"]]
st.line_chart(plot_test)

st.subheader("Forecast for the Next 12 Months")
st.line_chart(forecast.set_index("Date")["Forecast_Sales"])
st.dataframe(forecast, use_container_width=True)

with st.expander("Model evaluation and interpretation"):
    st.write("MAE is the average absolute prediction error; lower is generally better.")
    st.write("RMSE penalizes larger errors more strongly; lower is generally better.")
    st.write("R² compares model predictions with a simple mean-based baseline. It can be negative.")
    st.write(f"Test-period R²: {metrics.loc[0, 'R2']:.3f}")

st.caption("Educational example using a generated sample dataset. Forecasts are estimates, not guarantees.")
