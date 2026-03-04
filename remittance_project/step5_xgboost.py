"""
STEP 5 — XGBOOST MODEL
Pakistan Remittance Flow Predictor

XGBoost = Extreme Gradient Boosting
- Uses ALL engineered features (oil lags, PKR, Gulf employment etc.)
- Gives feature importance scores — tells us WHICH signals matter most
- Best overall accuracy on structured tabular time-series data
- Uses TimeSeriesSplit cross-validation (no data leakage)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── Load data ────────────────────────────────────────────────────
train = pd.read_csv("data/train.csv", index_col="date", parse_dates=True)
test  = pd.read_csv("data/test.csv",  index_col="date", parse_dates=True)

with open("data/feature_cols.txt") as f:
    FEATURE_COLS = [line.strip() for line in f.readlines()]

TARGET = "remittances_usd_mn"
X_train = train[FEATURE_COLS]
y_train = train[TARGET]
X_test  = test[FEATURE_COLS]
y_test  = test[TARGET]

print("=" * 60)
print("STEP 5: XGBoost Model")
print("=" * 60)
print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
print(f"Features: {len(FEATURE_COLS)}")

BLUE = "#1A5276"; RED = "#C0392B"; GRN = "#1E8449"; ORG = "#E67E22"

# ════════════════════════════════════════════════════════════════
# 1. FIT XGBOOST
# ════════════════════════════════════════════════════════════════
print("\n[1/4] Training XGBoost model...")
try:
    from xgboost import XGBRegressor
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0
    )

    # Time-series cross validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=tscv, scoring="neg_mean_absolute_error")
    print(f"   CV MAE scores: {[-round(s,1) for s in cv_scores]}")
    print(f"   Mean CV MAE: ${-cv_scores.mean():.1f}M ± ${cv_scores.std():.1f}M")

    model.fit(X_train, y_train)
    xgb_available = True

except ImportError:
    print("   XGBoost not installed. Run: pip install xgboost scikit-learn")
    print("   Using Random Forest as fallback...")
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        model = RandomForestRegressor(
            n_estimators=300, max_depth=6,
            random_state=42, n_jobs=-1
        )
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=tscv, scoring="neg_mean_absolute_error")
        print(f"   RF CV MAE: ${-cv_scores.mean():.1f}M")
        model.fit(X_train, y_train)
        xgb_available = True
    except ImportError:
        print("   scikit-learn also not installed.")
        xgb_available = False

# ════════════════════════════════════════════════════════════════
# 2. PREDICT & EVALUATE
# ════════════════════════════════════════════════════════════════
print("\n[2/4] Generating predictions...")

if xgb_available:
    forecast_mean = model.predict(X_test)
    actuals = y_test.values

    mae  = np.mean(np.abs(actuals - forecast_mean))
    rmse = np.sqrt(np.mean((actuals - forecast_mean)**2))
    mape = np.mean(np.abs((actuals - forecast_mean) / actuals)) * 100
    r2   = 1 - np.sum((actuals - forecast_mean)**2) / np.sum((actuals - actuals.mean())**2)

    print(f"\n   {'Metric':<20} {'Value':>10}")
    print(f"   {'-'*32}")
    print(f"   {'MAE (USD Mn)':<20} {mae:>10.2f}")
    print(f"   {'RMSE (USD Mn)':<20} {rmse:>10.2f}")
    print(f"   {'MAPE (%)':<20} {mape:>10.2f}")
    print(f"   {'R² Score':<20} {r2:>10.4f}")

    # ── Feature importance ─────────────────────────────────────
    print("\n[3/4] Computing feature importance...")
    try:
        importances = model.feature_importances_
    except AttributeError:
        importances = model.feature_importances_

    feat_imp = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=False)
    top15 = feat_imp.head(15)

    print("\n   TOP 15 MOST IMPORTANT FEATURES:")
    print(f"   {'Rank':<5} {'Feature':<30} {'Importance':>10}")
    print(f"   {'-'*48}")
    for rank, (feat, imp) in enumerate(top15.items(), 1):
        print(f"   {rank:<5} {feat:<30} {imp:>10.4f}")

    # Save results
    results = {
        "model": "XGBoost",
        "mae":   round(mae, 2),
        "rmse":  round(rmse, 2),
        "mape":  round(mape, 2),
        "r2":    round(r2, 4),
        "forecast":    forecast_mean.tolist(),
        "actuals":     actuals.tolist(),
        "test_dates":  test.index.strftime("%Y-%m-%d").tolist(),
        "feature_importance": feat_imp.head(15).to_dict()
    }
    with open("models/xgboost_results.json", "w") as f:
        json.dump(results, f, indent=2)

else:
    print("   Could not run model — please install xgboost or scikit-learn")

# ════════════════════════════════════════════════════════════════
# 3. PLOTS
# ════════════════════════════════════════════════════════════════
if xgb_available:
    print("\n[4/4] Generating plots...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    # — Plot 1: Forecast vs Actual ————————————————————————————
    ax = axes[0]
    ax.plot(train.index[-36:], train[TARGET].iloc[-36:],
            color=BLUE, linewidth=1.8, alpha=0.7, label="Training (last 3yr)")
    ax.plot(test.index, actuals, color=GRN,
            linewidth=2.2, label="Actual", zorder=3)
    ax.plot(test.index, forecast_mean, color=RED,
            linewidth=2, linestyle="--", label="XGBoost Forecast", zorder=4)
    ax.fill_between(test.index,
                    forecast_mean * 0.93, forecast_mean * 1.07,
                    color=RED, alpha=0.1, label="±7% Band")
    ax.axvline(test.index[0], color="grey", linewidth=1.2, linestyle=":")
    ax.set_title("XGBoost Forecast vs Actual — Pakistan Remittances",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("USD Million", fontsize=11)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
    ax.grid(True, alpha=0.3)
    for s in ["top","right"]: ax.spines[s].set_visible(False)

    # — Plot 2: Feature Importance ————————————————————————————
    ax2 = axes[1]
    colors_fi = [RED if i < 3 else BLUE for i in range(len(top15))]
    bars = ax2.barh(range(len(top15)), top15.values[::-1],
                    color=colors_fi[::-1], alpha=0.85, edgecolor="white")
    ax2.set_yticks(range(len(top15)))
    ax2.set_yticklabels(top15.index[::-1], fontsize=9)
    ax2.set_xlabel("Feature Importance Score", fontsize=11)
    ax2.set_title("XGBoost Feature Importance — Top 15 Predictors\n"
                  "(Red = Top 3 most influential signals)",
                  fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")
    for s in ["top","right"]: ax2.spines[s].set_visible(False)

    # — Plot 3: Residuals ————————————————————————————————————
    ax3 = axes[2]
    residuals = actuals - forecast_mean
    colors_res = [GRN if r >= 0 else RED for r in residuals]
    ax3.bar(test.index, residuals, color=colors_res,
            alpha=0.75, width=20, edgecolor="white")
    ax3.axhline(0, color="black", linewidth=1)
    ax3.axhline( mae, color=RED, linewidth=1.5, linestyle="--", label=f"+MAE ${mae:.0f}M")
    ax3.axhline(-mae, color=RED, linewidth=1.5, linestyle="--", label=f"-MAE ${mae:.0f}M")
    ax3.set_title(f"Residuals  |  MAE=${mae:.0f}M  |  MAPE={mape:.1f}%  |  R²={r2:.4f}",
                  fontsize=11, fontweight="bold")
    ax3.set_ylabel("Forecast Error (USD Mn)", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
    ax3.grid(True, alpha=0.3)
    for s in ["top","right"]: ax3.spines[s].set_visible(False)

    plt.tight_layout()
    plt.savefig("outputs/08_xgboost_results.png", bbox_inches="tight")
    plt.close()
    print("   Saved: outputs/08_xgboost_results.png")
    print("   Saved: models/xgboost_results.json")
    print(f"\n✅ XGBoost complete → MAE: ${mae:.0f}M | MAPE: {mape:.1f}% | R²: {r2:.4f}")
