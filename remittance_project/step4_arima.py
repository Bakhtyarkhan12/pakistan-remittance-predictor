"""
STEP 4 — ARIMA BASELINE MODEL
Pakistan Remittance Flow Predictor

ARIMA = AutoRegressive Integrated Moving Average
- Univariate: uses ONLY past remittance values
- No external features (oil, PKR, etc.)
- Purpose: establish a baseline to beat with ML models
- If ARIMA is nearly as good as XGBoost/LSTM, external
  features don't add much value (they do here)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import pickle
import os

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── Load data ────────────────────────────────────────────────────
train = pd.read_csv("data/train.csv", index_col="date", parse_dates=True)
test  = pd.read_csv("data/test.csv",  index_col="date", parse_dates=True)
TARGET = "remittances_usd_mn"

print("=" * 60)
print("STEP 4: ARIMA Baseline Model")
print("=" * 60)
print(f"Train: {len(train)} months | Test: {len(test)} months")

# ════════════════════════════════════════════════════════════════
# 1. TEST STATIONARITY — ARIMA needs stationary time series
# ════════════════════════════════════════════════════════════════
print("\n[1/4] Testing stationarity (ADF Test)...")

try:
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(train[TARGET], autolag="AIC")
    print(f"   ADF Statistic : {adf_result[0]:.4f}")
    print(f"   p-value       : {adf_result[1]:.4f}")
    print(f"   Stationary?   : {'YES ✅' if adf_result[1] < 0.05 else 'NO — differencing needed'}")
    stationary = adf_result[1] < 0.05
except ImportError:
    print("   statsmodels not installed — skipping ADF test")
    print("   Run: pip install statsmodels")
    stationary = False

# ════════════════════════════════════════════════════════════════
# 2. FIT ARIMA MODEL
# ════════════════════════════════════════════════════════════════
print("\n[2/4] Fitting ARIMA model...")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # d=1 because remittances have a clear upward trend (non-stationary)
    # p=2 autoregressive terms, q=2 moving average terms
    # Seasonal: P=1, D=1, Q=1, s=12 (monthly seasonality)
    print("   Fitting SARIMA(2,1,2)(1,1,1,12) — includes seasonal component...")

    model = SARIMAX(
        train[TARGET],
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted = model.fit(disp=False)
    print(f"   AIC: {fitted.aic:.2f}")
    print(f"   BIC: {fitted.bic:.2f}")
    arima_available = True

except ImportError:
    print("   statsmodels not installed.")
    print("   Using naive forecast as fallback (12-month seasonal naive)")
    arima_available = False

# ════════════════════════════════════════════════════════════════
# 3. GENERATE FORECASTS
# ════════════════════════════════════════════════════════════════
print("\n[3/4] Generating forecasts on test set...")

if arima_available:
    forecast_result = fitted.get_forecast(steps=len(test))
    forecast_mean   = forecast_result.predicted_mean.values
    conf_int        = forecast_result.conf_int()
    lower           = conf_int.iloc[:, 0].values
    upper           = conf_int.iloc[:, 1].values
else:
    # Seasonal naive: use value from same month 1 year ago
    combined = pd.concat([train, test])
    forecast_mean = []
    for i in range(len(test)):
        idx = test.index[i]
        try:
            naive_val = train[TARGET].loc[
                train.index.month == idx.month
            ].iloc[-1]
        except IndexError:
            naive_val = train[TARGET].mean()
        forecast_mean.append(naive_val)
    forecast_mean = np.array(forecast_mean)
    lower = forecast_mean * 0.85
    upper = forecast_mean * 1.15

# ════════════════════════════════════════════════════════════════
# 4. EVALUATE PERFORMANCE
# ════════════════════════════════════════════════════════════════
print("\n[4/4] Evaluating model performance...")
actuals = test[TARGET].values

mae  = np.mean(np.abs(actuals - forecast_mean))
rmse = np.sqrt(np.mean((actuals - forecast_mean) ** 2))
mape = np.mean(np.abs((actuals - forecast_mean) / actuals)) * 100
r2   = 1 - np.sum((actuals - forecast_mean)**2) / np.sum((actuals - actuals.mean())**2)

print(f"\n   {'Metric':<20} {'Value':>10}")
print(f"   {'-'*32}")
print(f"   {'MAE (USD Mn)':<20} {mae:>10.2f}")
print(f"   {'RMSE (USD Mn)':<20} {rmse:>10.2f}")
print(f"   {'MAPE (%)':<20} {mape:>10.2f}")
print(f"   {'R² Score':<20} {r2:>10.4f}")

# Save results
arima_results = {
    "model": "ARIMA/SARIMA",
    "mae":   round(mae, 2),
    "rmse":  round(rmse, 2),
    "mape":  round(mape, 2),
    "r2":    round(r2, 4),
    "forecast": forecast_mean.tolist(),
    "lower": lower.tolist(),
    "upper": upper.tolist(),
    "test_dates": test.index.strftime("%Y-%m-%d").tolist(),
    "actuals": actuals.tolist()
}

import json
with open("models/arima_results.json", "w") as f:
    json.dump(arima_results, f, indent=2)

# ── Plot ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
BLUE = "#1A5276"; RED = "#C0392B"; GRN = "#1E8449"

# Top: Full timeline with forecast
ax = axes[0]
ax.plot(train.index, train[TARGET], color=BLUE,
        linewidth=1.8, label="Training Data", alpha=0.9)
ax.plot(test.index, actuals, color=GRN,
        linewidth=2, label="Actual (Test Period)", alpha=0.9)
ax.plot(test.index, forecast_mean, color=RED,
        linewidth=2, linestyle="--", label="ARIMA Forecast")
ax.fill_between(test.index, lower, upper,
                color=RED, alpha=0.12, label="95% Confidence Interval")
ax.axvline(test.index[0], color="grey", linewidth=1.2,
           linestyle=":", alpha=0.8)
ax.text(test.index[0], ax.get_ylim()[0] * 1.01,
        " Test Period →", fontsize=9, color="grey")
ax.set_title("ARIMA Baseline Forecast — Pakistan Remittances",
             fontsize=13, fontweight="bold")
ax.set_ylabel("USD Million", fontsize=11)
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
ax.grid(True, alpha=0.3)
for spine in ["top","right"]:
    ax.spines[spine].set_visible(False)

# Bottom: Residuals
residuals = actuals - forecast_mean
ax2 = axes[1]
colors_res = [RED if r < 0 else GRN for r in residuals]
ax2.bar(test.index, residuals, color=colors_res, alpha=0.7,
        width=20, edgecolor="white")
ax2.axhline(0, color="black", linewidth=1)
ax2.axhline(mae, color=RED, linewidth=1, linestyle="--",
            label=f"+MAE (${mae:.0f}M)")
ax2.axhline(-mae, color=RED, linewidth=1, linestyle="--",
            label=f"-MAE (${mae:.0f}M)")
ax2.set_title(f"Forecast Residuals  |  MAE=${mae:.0f}M  |  MAPE={mape:.1f}%",
              fontsize=11, fontweight="bold")
ax2.set_ylabel("Residual (USD Mn)", fontsize=11)
ax2.legend(fontsize=9)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
ax2.grid(True, alpha=0.3)
for spine in ["top","right"]:
    ax2.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/07_arima_forecast.png", bbox_inches="tight")
plt.close()

print("\n✅ ARIMA complete!")
print("   Saved: outputs/07_arima_forecast.png")
print("   Saved: models/arima_results.json")
print(f"\n   ARIMA Benchmark → MAE: ${mae:.0f}M | MAPE: {mape:.1f}% | R²: {r2:.4f}")
print("   (ML models should beat this — otherwise features add no value)")
