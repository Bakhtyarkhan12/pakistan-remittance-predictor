"""
STEP 6 — LSTM MODEL (with Exponential Smoothing fallback)
Pakistan Remittance Flow Predictor

If TensorFlow is installed:
  → Trains a Bidirectional LSTM on 12-month sliding windows

If TensorFlow is NOT installed:
  → Uses Holt-Winters Triple Exponential Smoothing (scipy)
    which captures: level + trend + 12-month seasonality

Install for full LSTM:
  pip install tensorflow scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, os, warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

train = pd.read_csv("data/train.csv", index_col="date", parse_dates=True)
test  = pd.read_csv("data/test.csv",  index_col="date", parse_dates=True)

with open("data/feature_cols.txt") as f:
    FEATURE_COLS = [l.strip() for l in f.readlines()]

TARGET = "remittances_usd_mn"
BLUE = "#1A5276"; RED = "#C0392B"; GRN = "#1E8449"; PURPLE = "#6C3483"
WINDOW = 12

print("=" * 60)
print("STEP 6: LSTM / Sequence Model")
print("=" * 60)

lstm_available = False
history_data   = None
model_label    = ""

# ════════════════════════════════════════════════════════════════
# TRY TENSORFLOW FIRST
# ════════════════════════════════════════════════════════════════
try:
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam

    print("\nTensorFlow found — running full LSTM")
    tf.random.set_seed(42)

    fs = MinMaxScaler(); ts = MinMaxScaler()
    Xtr = fs.fit_transform(train[FEATURE_COLS].values)
    Xte = fs.transform(test[FEATURE_COLS].values)
    ytr = ts.fit_transform(train[TARGET].values.reshape(-1,1))
    yte = ts.transform(test[TARGET].values.reshape(-1,1))

    def make_seq(X, y, w):
        Xs, ys = [], []
        for i in range(w, len(X)):
            Xs.append(X[i-w:i]); ys.append(y[i])
        return np.array(Xs), np.array(ys)

    Xa = np.vstack([Xtr, Xte]); ya = np.vstack([ytr, yte])
    Xseq, yseq = make_seq(Xa, ya, WINDOW)
    n = len(Xtr) - WINDOW
    Xts, yts = Xseq[n:], yseq[n:]
    Xtr2, ytr2 = Xseq[:n], yseq[:n]

    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True,
                           input_shape=(WINDOW, Xtr2.shape[2]))),
        Dropout(0.2), LSTM(32), Dropout(0.15),
        Dense(16, activation="relu"), Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss="huber", metrics=["mae"])
    print("[Training LSTM — 1-2 minutes...]")
    hist = model.fit(Xtr2, ytr2, epochs=150, batch_size=16,
                     validation_split=0.15, verbose=0,
                     callbacks=[EarlyStopping(patience=15,
                                              restore_best_weights=True),
                                ReduceLROnPlateau(factor=0.5, patience=7)])
    print(f"   Trained {len(hist.history['loss'])} epochs")

    pred = model.predict(Xts, verbose=0)
    forecast_mean = ts.inverse_transform(pred).flatten()
    actuals       = ts.inverse_transform(yts).flatten()
    # Align all three to same length to prevent shape mismatch in plots
    n_out         = min(len(forecast_mean), len(actuals), len(test.index) - WINDOW)
    forecast_mean = forecast_mean[:n_out]
    actuals       = actuals[:n_out]
    test_dates    = test.index[WINDOW:WINDOW + n_out]
    model_label   = "LSTM (Bidirectional)"
    history_data  = hist.history
    lstm_available = True

except ImportError:
    print("\nTensorFlow not installed — using Holt-Winters Exponential Smoothing")
    print("(pip install tensorflow scikit-learn for full LSTM)\n")

    from scipy.optimize import minimize

    def holtwinters(series, alpha, beta, gamma, h, sp=12):
        n = len(series)
        level = np.mean(series[:sp])
        trend = (np.mean(series[sp:2*sp]) - np.mean(series[:sp])) / sp
        season = [series[i] - level for i in range(sp)]
        fitted = []
        for i in range(n):
            m = i % sp
            pl, pt = level, trend
            level  = alpha*(series[i]-season[m]) + (1-alpha)*(pl+pt)
            trend  = beta*(level-pl) + (1-beta)*pt
            season[m] = gamma*(series[i]-level) + (1-gamma)*season[m]
            fitted.append(pl + pt + season[m])
        fc = [level + j*trend + season[(n+j-1)%sp] for j in range(1, h+1)]
        return np.array(fitted), np.array(fc)

    def sse(p, s):
        a, b, g = np.clip(p, 0.01, 0.99)
        try:
            f, _ = holtwinters(s, a, b, g, h=1)
            return float(np.sum((s - f)**2))
        except Exception:
            return 1e12

    ts = train[TARGET].values
    print("   Optimising smoothing parameters...")
    res = minimize(sse, x0=[0.3, 0.1, 0.3], args=(ts,),
                   method="Nelder-Mead",
                   options={"maxiter": 3000, "xatol": 1e-7})
    a, b, g = np.clip(res.x, 0.01, 0.99)
    print(f"   α={a:.3f}  β={b:.3f}  γ={g:.3f}")

    _, fc = holtwinters(ts, a, b, g, h=len(test))
    forecast_mean = fc[:len(test)]
    actuals       = test[TARGET].values[:len(forecast_mean)]
    test_dates    = test.index[:len(forecast_mean)]
    model_label   = "Holt-Winters (Triple Exponential Smoothing)"
    lstm_available = True

# ════════════════════════════════════════════════════════════════
# EVALUATE
# ════════════════════════════════════════════════════════════════
mae  = np.mean(np.abs(actuals - forecast_mean))
rmse = np.sqrt(np.mean((actuals - forecast_mean)**2))
mape = np.mean(np.abs((actuals - forecast_mean) / actuals)) * 100
r2   = 1 - np.sum((actuals-forecast_mean)**2) / np.sum((actuals-actuals.mean())**2)

print(f"\n   {'Metric':<20} {'Value':>10}")
print(f"   {'-'*32}")
print(f"   {'MAE (USD Mn)':<20} {mae:>10.2f}")
print(f"   {'RMSE (USD Mn)':<20} {rmse:>10.2f}")
print(f"   {'MAPE (%)':<20} {mape:>10.2f}")
print(f"   {'R² Score':<20} {r2:>10.4f}")

with open("models/lstm_results.json", "w") as f:
    json.dump({
        "model": model_label,
        "mae": round(mae,2), "rmse": round(rmse,2),
        "mape": round(mape,2), "r2": round(r2,4),
        "forecast":   forecast_mean.tolist(),
        "actuals":    actuals.tolist(),
        "test_dates": pd.DatetimeIndex(test_dates).strftime("%Y-%m-%d").tolist()
    }, f, indent=2)

# ════════════════════════════════════════════════════════════════
# PLOTS
# ════════════════════════════════════════════════════════════════
n_panels = 3 if history_data else 2
fig, axes = plt.subplots(n_panels, 1, figsize=(14, 5*n_panels))
axlist = list(axes) if n_panels > 1 else [axes]
pi = 0

if history_data:
    ax = axlist[pi]; pi += 1
    ax.plot(history_data["loss"],     color=BLUE, lw=2, label="Train Loss")
    ax.plot(history_data["val_loss"], color=RED,  lw=2, ls="--", label="Val Loss")
    ax.set_title("LSTM Training Loss per Epoch", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    for s in ["top","right"]: ax.spines[s].set_visible(False)

ax = axlist[pi]; pi += 1
ax.plot(train.index[-36:], train[TARGET].iloc[-36:],
        color=BLUE, lw=1.8, alpha=0.6, label="Training (last 3yr)")
ax.plot(test_dates, actuals,       color=GRN,    lw=2.2, label="Actual")
ax.plot(test_dates, forecast_mean, color=PURPLE, lw=2,
        ls="--", label=f"{model_label}")
ax.fill_between(test_dates, forecast_mean*0.93, forecast_mean*1.07,
                color=PURPLE, alpha=0.1, label="±7% Band")
ax.axvline(test_dates[0], color="grey", lw=1.2, ls=":")
ax.set_title(f"{model_label} — Forecast vs Actual", fontsize=12, fontweight="bold")
ax.set_ylabel("USD Million"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}M"))
for s in ["top","right"]: ax.spines[s].set_visible(False)

ax = axlist[pi]
res = actuals - forecast_mean
cols = [GRN if r >= 0 else RED for r in res]
ax.bar(range(len(test_dates)), res, color=cols, alpha=0.75, width=0.8, edgecolor="white")
ax.set_xticks(range(0, len(test_dates), max(1, len(test_dates)//8)))
ax.set_xticklabels([str(test_dates[i])[:7] for i in range(0, len(test_dates), max(1, len(test_dates)//8))], rotation=30)
ax.axhline(0,    color="black", lw=1)
ax.axhline( mae, color=RED, lw=1.5, ls="--", label=f"±MAE ${mae:.0f}M")
ax.axhline(-mae, color=RED, lw=1.5, ls="--")
ax.set_title(f"Residuals | MAE=${mae:.0f}M | MAPE={mape:.1f}% | R²={r2:.4f}",
             fontsize=11, fontweight="bold")
ax.set_ylabel("Forecast Error (USD Mn)"); ax.legend(); ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}M"))
for s in ["top","right"]: ax.spines[s].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/09_lstm_results.png", bbox_inches="tight")
plt.close()

print(f"\n✅ {model_label} complete!")
print(f"   MAE: ${mae:.0f}M | MAPE: {mape:.1f}% | R²: {r2:.4f}")
print("   Saved: outputs/09_lstm_results.png")
print("   Saved: models/lstm_results.json")