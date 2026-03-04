"""
STEP 3 — FEATURE ENGINEERING
Pakistan Remittance Flow Predictor

Transforms raw data into ML-ready features:
- Lag features (oil price delayed effects)
- Rolling statistics (volatility, momentum)
- Seasonal dummies
- Percentage changes
- Train/test split (80/20 chronological)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/raw_dataset.csv", index_col="date", parse_dates=True)

print("=" * 60)
print("STEP 3: Feature Engineering")
print("=" * 60)

# ════════════════════════════════════════════════════════════════
# 1. LAG FEATURES — oil price effect is delayed
# ════════════════════════════════════════════════════════════════
print("\n[1/4] Creating lag features...")
for lag in [1, 2, 3, 6]:
    df[f"oil_lag_{lag}m"]    = df["oil_price_brent"].shift(lag)
    df[f"remit_lag_{lag}m"]  = df["remittances_usd_mn"].shift(lag)

df["gulf_lag_2m"] = df["gulf_employment_idx"].shift(2)
df["pkr_lag_1m"]  = df["usd_pkr"].shift(1)
print("   Created: oil lags (1,2,3,6m), remittance lags (1,2,3,6m), gulf lag 2m, pkr lag 1m")

# ════════════════════════════════════════════════════════════════
# 2. ROLLING STATISTICS — capture trends and volatility
# ════════════════════════════════════════════════════════════════
print("\n[2/4] Creating rolling statistics...")
df["oil_3m_avg"]       = df["oil_price_brent"].rolling(3).mean()
df["oil_6m_avg"]       = df["oil_price_brent"].rolling(6).mean()
df["pkr_3m_volatility"]= df["usd_pkr"].rolling(3).std()
df["pkr_6m_avg"]       = df["usd_pkr"].rolling(6).mean()
df["remit_3m_avg"]     = df["remittances_usd_mn"].rolling(3).mean()
df["remit_6m_avg"]     = df["remittances_usd_mn"].rolling(6).mean()
df["gulf_3m_avg"]      = df["gulf_employment_idx"].rolling(3).mean()
print("   Created: rolling means (3m, 6m) and PKR volatility")

# ════════════════════════════════════════════════════════════════
# 3. PERCENTAGE CHANGES — momentum signals
# ════════════════════════════════════════════════════════════════
print("\n[3/4] Creating momentum features...")
df["oil_mom_1m"]   = df["oil_price_brent"].pct_change(1)
df["oil_mom_3m"]   = df["oil_price_brent"].pct_change(3)
df["pkr_mom_1m"]   = df["usd_pkr"].pct_change(1)
df["remit_mom_1m"] = df["remittances_usd_mn"].pct_change(1)

# Oil price direction: rising vs falling (binary)
df["oil_rising"]   = (df["oil_mom_1m"] > 0).astype(int)
df["pkr_depreciating"] = (df["pkr_mom_1m"] > 0).astype(int)
print("   Created: % change momentum + direction signals")

# ════════════════════════════════════════════════════════════════
# 4. SEASONAL & CALENDAR FEATURES
# ════════════════════════════════════════════════════════════════
print("\n[4/4] Creating seasonal features...")
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["quarter"]   = ((df["month"] - 1) // 3) + 1
df["is_q4"]     = (df["quarter"] == 4).astype(int)

# Approximate Eid months (major remittance boost)
# Eid moves ~11 days earlier each year in Gregorian calendar
eid_months_by_year = {
    2010: [9], 2011: [8], 2012: [8], 2013: [8],
    2014: [7], 2015: [7], 2016: [7], 2017: [6],
    2018: [6], 2019: [6], 2020: [5], 2021: [5],
    2022: [5], 2023: [4], 2024: [4]
}
df["is_eid_month"] = 0
for idx in df.index:
    yr  = idx.year
    mon = idx.month
    if yr in eid_months_by_year and mon in eid_months_by_year[yr]:
        df.loc[idx, "is_eid_month"] = 1

df["post_covid"]  = (df.index >= "2020-06-01").astype(int)
df["crisis_2022"] = ((df.index >= "2022-04-01") &
                     (df.index <= "2022-10-01")).astype(int)
print("   Created: month sin/cos, quarter, Eid dummy, post-COVID, crisis dummies")

# ════════════════════════════════════════════════════════════════
# DROP ROWS WITH NaN (from lags/rolling — first ~6 months)
# ════════════════════════════════════════════════════════════════
df_clean = df.dropna()
print(f"\n   Rows before cleaning: {len(df)}")
print(f"   Rows after dropping NaN: {len(df_clean)}")
print(f"   Features created: {len(df_clean.columns)} total columns")

# ════════════════════════════════════════════════════════════════
# TRAIN / TEST SPLIT — must be chronological (no random shuffle!)
# 80% train (≈ 2010–2021), 20% test (≈ 2022–2024)
# ════════════════════════════════════════════════════════════════
split_idx  = int(len(df_clean) * 0.80)
train_df   = df_clean.iloc[:split_idx]
test_df    = df_clean.iloc[split_idx:]

print(f"\n   Train period: {train_df.index[0].strftime('%b %Y')} → {train_df.index[-1].strftime('%b %Y')} ({len(train_df)} months)")
print(f"   Test period:  {test_df.index[0].strftime('%b %Y')} → {test_df.index[-1].strftime('%b %Y')} ({len(test_df)} months)")

# ════════════════════════════════════════════════════════════════
# DEFINE FEATURE COLUMNS
# ════════════════════════════════════════════════════════════════
TARGET = "remittances_usd_mn"

FEATURE_COLS = [
    # Raw signals
    "oil_price_brent", "usd_pkr", "gulf_employment_idx", "pol_stability_idx",
    # Lag features
    "oil_lag_1m", "oil_lag_2m", "oil_lag_3m", "oil_lag_6m",
    "remit_lag_1m", "remit_lag_2m", "remit_lag_3m",
    "gulf_lag_2m", "pkr_lag_1m",
    # Rolling stats
    "oil_3m_avg", "oil_6m_avg",
    "pkr_3m_volatility", "pkr_6m_avg",
    "gulf_3m_avg",
    # Momentum
    "oil_mom_1m", "oil_mom_3m", "pkr_mom_1m",
    "oil_rising", "pkr_depreciating",
    # Seasonal
    "month_sin", "month_cos", "is_eid_month", "is_q4",
    "post_covid", "crisis_2022",
]

# Verify all columns exist
missing = [c for c in FEATURE_COLS if c not in df_clean.columns]
if missing:
    print(f"\n⚠️  Missing columns: {missing}")
else:
    print(f"\n   All {len(FEATURE_COLS)} feature columns verified ✅")

# ════════════════════════════════════════════════════════════════
# SAVE PROCESSED DATA
# ════════════════════════════════════════════════════════════════
df_clean.to_csv("data/engineered_dataset.csv")
train_df.to_csv("data/train.csv")
test_df.to_csv("data/test.csv")

# Save feature list
with open("data/feature_cols.txt", "w") as f:
    f.write("\n".join(FEATURE_COLS))

print("\n✅ Feature engineering complete!")
print("   Saved: data/engineered_dataset.csv")
print("   Saved: data/train.csv")
print("   Saved: data/test.csv")
print("   Saved: data/feature_cols.txt")

# Quick feature summary plot
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
plots = [
    ("oil_lag_3m", "Oil Price (3-Month Lag)", "#C0392B"),
    ("pkr_3m_volatility", "PKR 3-Month Volatility", "#D4AC0D"),
    ("remit_lag_1m", "Remittances (1-Month Lag)", "#1A5276"),
    ("gulf_3m_avg", "Gulf Employment (3M Avg)", "#1E8449"),
]
for ax, (col, title, color) in zip(axes.flat, plots):
    ax.scatter(df_clean[col], df_clean[TARGET],
               alpha=0.4, color=color, s=20, edgecolors="none")
    z = np.polyfit(df_clean[col].dropna(),
                   df_clean.loc[df_clean[col].notna(), TARGET], 1)
    p = np.poly1d(z)
    xr = np.linspace(df_clean[col].min(), df_clean[col].max(), 100)
    ax.plot(xr, p(xr), color="black", linewidth=1.5, linestyle="--")
    corr = df_clean[col].corr(df_clean[TARGET])
    ax.set_title(f"{title}\nr = {corr:.3f}", fontsize=10, fontweight="bold")
    ax.set_xlabel(col, fontsize=9)
    ax.set_ylabel("Remittances (USD Mn)", fontsize=9)
    ax.grid(True, alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

plt.suptitle("Key Features vs Remittances — Scatter Plots",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("outputs/06_feature_scatter.png", bbox_inches="tight")
plt.close()
print("   Saved: outputs/06_feature_scatter.png")
