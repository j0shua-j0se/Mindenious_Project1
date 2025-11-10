# 📈 Walmart Sales Forecasting (Time Series)

Predict weekly sales for Walmart stores using historical data and machine learning. The project engineers time-series features (lags, rolling means, seasonality) and compares Linear Regression, Random Forest, and XGBoost. Best model (XGBoost) achieved approx. R² ≈ 0.961 with strong error reduction on the holdout period.

---

## 1) Quick Start

### A. Setup
```
# clone your repo (replace with your URL)
git clone https://github.com/<your-username>/walmart-sales-forecasting.git
cd walmart-sales-forecasting

# create env and install deps
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

### B. Run (choose one)
```
# Option 1: Notebook
jupyter notebook notebooks/sales_forecasting.ipynb

# Option 2: Script (training + evaluation)
python src/train_model.py --data data/Walmart.csv --model xgboost --save-model models/xgb.pkl

# Option 3: Predict with a saved model
python src/predict.py --model models/xgb.pkl --data data/Walmart.csv --out outputs/predictions.csv
```

---

## 2) Data

- File: `data/Walmart.csv`  
- Columns: `Store, Date, Weekly_Sales, Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment`  
- Frequency: weekly; multiple stores (IDs 1–45).  
- Target: `Weekly_Sales`.

---

## 3) Approach (one glance)

- Preprocess: parse `Date`, sort by `Store, Date`.
- Features:
  - Lags: `Sales_Lag_1/2/3/6`
  - Rolling means: `Sales_Roll_Mean_3/6`
  - Seasonality: `Year, Month, Week, DayOfYear`
  - Exogenous: `Holiday_Flag, Temperature, Fuel_Price, CPI, Unemployment`, `Store`
- Split: time-ordered 80/20 (no shuffle).
- Models: Linear Regression, Random Forest, XGBoost.
- Metrics: MAE, RMSE, R².

Approximate benchmark (holdout):
- Linear Regression: MAE ~ 48k | RMSE ~ 82k | R² ~ 0.93  
- Random Forest: MAE ~ 42k | RMSE ~ 71k | R² ~ 0.95  
- XGBoost (best): MAE ~ 39.9k | RMSE ~ 68.5k | R² ~ 0.961

---

## 4) Project Layout (minimal)

```
.
├── data/
│   └── Walmart.csv
├── notebooks/
│   └── sales_forecasting.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py              # wrappers for LR/RF/XGB
│   ├── train_model.py        # CLI for training/eval
│   └── predict.py            # CLI for batch prediction
├── models/
├── outputs/
│   ├── predictions.csv
│   └── visualizations/       # trend, feature importance, actual vs predicted
├── requirements.txt
└── README.md
```

---

## 5) Reproduce Figures (optional)

- Sales trend: `outputs/visualizations/sales_trend.png`
- Actual vs Predicted (XGB): `outputs/visualizations/actual_vs_predicted_xgb.png`
- Feature importance: `outputs/visualizations/feature_importance.png`
- Model comparison: `outputs/visualizations/model_comparison.png`

---

## 6) Configuration (common flags)

- `--model {linear,rf,xgboost}` (default: xgboost)  
- `--test-size 0.2` (time-based split)  
- `--random-state 42`  
- `--save-model models/xgb.pkl`

---

## 7) Requirements

```
pandas>=1.3
numpy>=1.21
scikit-learn>=1.0
xgboost>=1.5
matplotlib>=3.4
seaborn>=0.11
jupyter>=1.0
```

Install all via:
```
pip install -r requirements.txt
```

---

## 8) Notes

- Avoid data leakage: never shuffle time series before splitting.
- Holiday spikes may need holiday-specific models for best accuracy.
- Extend easily to multi-step horizons or store-level specialized models.
