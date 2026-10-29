# Predictive Analytics Using Historical Data

## Overview
A Python project that cleans historical monthly sales data, identifies a trend, trains a regression model, evaluates predictions, and forecasts the next 12 months. It also includes an optional Streamlit dashboard.

## Features
- Parses dates and converts sales to numeric values
- Removes invalid rows and duplicate dates
- Creates time and seasonal features
- Uses a chronological train/test split to avoid training on future observations
- Trains a Linear Regression model
- Evaluates with MAE, RMSE, and R²
- Saves actual-vs-predicted and future forecast charts
- Exports forecasts and test predictions to CSV
- Optional Streamlit dashboard

## Project Structure
```text
Predictive_Analytics_Project/
├── app.py
├── predictive_analytics.py
├── requirements.txt
├── README.md
├── data/
│   └── historical_sales.csv
└── outputs/
    └── (created when the model runs)
```

## Requirements
Python 3.10 or newer is recommended.

## Run in VS Code
1. Extract the project ZIP.
2. Open the extracted folder in VS Code.
3. Open a terminal in that folder.
4. Install packages:
   ```bash
   python -m pip install -r requirements.txt
   ```
5. Run the model:
   ```bash
   python predictive_analytics.py
   ```
6. View the generated files in `outputs/`.
7. To launch the dashboard:
   ```bash
   streamlit run app.py
   ```

## Dataset
The included CSV is a reproducible **synthetic educational dataset** with monthly sales values. Replace `data/historical_sales.csv` with your own real historical data, keeping the column names `Date` and `Sales`, to analyze a real use case.

## Evaluation
- **MAE:** average absolute error in sales units.
- **RMSE:** square root of average squared error; larger errors have more impact.
- **R²:** comparison with a mean-based baseline; negative values are possible.

## Limitations
This is a baseline model, not a guarantee of future sales. It uses a simple linear trend and month seasonality. Real-world forecasts may need more history, holidays, promotions, prices, external variables, and comparisons with seasonal-naive or ARIMA models. Forecasts can be inaccurate when patterns change.

## Learning Outcomes
Historical-data cleaning, exploratory trend analysis, regression, chronological validation, forecast evaluation, visualization, and dashboard development.
