# Pakistan Remittance Flow Predictor — Setup Guide

## Install Required Packages
Run this once in your terminal:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels tensorflow openpyxl requests jupyter plotly
```

## Project Structure
```
remittance_project/
├── data/                   ← raw downloaded data goes here
├── outputs/                ← charts, results saved here
├── models/                 ← saved model files
├── step1_collect_data.py   ← downloads & builds dataset
├── step2_eda.py            ← exploratory data analysis + charts
├── step3_features.py       ← feature engineering
├── step4_arima.py          ← ARIMA baseline model
├── step5_xgboost.py        ← XGBoost model + feature importance
├── step6_lstm.py           ← LSTM deep learning model
├── step7_compare.py        ← final comparison + report
└── SETUP.md                ← this file
```

## Run Order
```bash
python step1_collect_data.py
python step2_eda.py
python step3_features.py
python step4_arima.py
python step5_xgboost.py
python step6_lstm.py
python step7_compare.py
```
