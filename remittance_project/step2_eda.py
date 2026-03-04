"""
STEP 2 — EXPLORATORY DATA ANALYSIS (EDA)
Pakistan Remittance Flow Predictor

Generates 5 publication-quality charts:
  1. Remittance trend over time with key events annotated
  2. All features plotted together
  3. Correlation heatmap
  4. Cross-correlation: oil price lag vs remittances (KEY FINDING)
  5. Seasonal pattern by month
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True)

# ── Load data ────────────────────────────────────────────────────
df = pd.read_csv("data/raw_dataset.csv", index_col="date", parse_dates=True)
print("=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)
print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ── Style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})
BLUE = "#1A5276"
RED  = "#C0392B"
GOLD = "#D4AC0D"
GRN  = "#1E8449"

# ════════════════════════════════════════════════════════════════
# CHART 1 — Remittance Trend + Events
# ════════════════════════════════════════════════════════════════
print("[1/5] Plotting remittance trend...")
fig, ax = plt.subplots(figsize=(14, 5))

ax.fill_between(df.index, df["remittances_usd_mn"],
                alpha=0.15, color=BLUE)
ax.plot(df.index, df["remittances_usd_mn"],
        color=BLUE, linewidth=2, label="Monthly Remittances (USD Mn)")

# 12-month rolling average
roll = df["remittances_usd_mn"].rolling(12).mean()
ax.plot(df.index, roll, color=RED, linewidth=2.5,
        linestyle="--", label="12-Month Rolling Average")

# Annotate key events
events = {
    "2015-01-01": ("Oil Price\nCrash", -280),
    "2020-04-01": ("COVID-19\nLockdowns", -320),
    "2021-06-01": ("COVID\nRecovery", 220),
    "2022-04-01": ("Pakistan\nPolitical Crisis", -300),
}
for date_str, (label, offset) in events.items():
    dt = pd.Timestamp(date_str)
    val = df.loc[dt, "remittances_usd_mn"] if dt in df.index else df["remittances_usd_mn"].mean()
    ax.annotate(label, xy=(dt, val), xytext=(dt, val + offset),
                fontsize=8, color="#555555", ha="center",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=1),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", alpha=0.8))

ax.set_title("Pakistan Workers' Remittances — Monthly Inflows (2010–2024)",
             fontsize=14, fontweight="bold", pad=15)
ax.set_ylabel("USD Million", fontsize=11)
ax.set_xlabel("")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.legend(fontsize=10)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
plt.tight_layout()
plt.savefig("outputs/01_remittance_trend.png", bbox_inches="tight")
plt.close()
print("   Saved: outputs/01_remittance_trend.png")

# ════════════════════════════════════════════════════════════════
# CHART 2 — All Features Dashboard
# ════════════════════════════════════════════════════════════════
print("[2/5] Plotting all features dashboard...")
fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

plot_cfg = [
    ("remittances_usd_mn", "Remittances (USD Mn)",     BLUE,  "${x:,.0f}M"),
    ("oil_price_brent",    "Brent Crude (USD/barrel)", RED,   "${x:.0f}"),
    ("usd_pkr",            "USD/PKR Exchange Rate",    GOLD,  "{x:.0f}"),
    ("gulf_employment_idx","Gulf Employment Index",    GRN,   "{x:.0f}"),
]

for ax, (col, title, color, fmt) in zip(axes, plot_cfg):
    ax.plot(df.index, df[col], color=color, linewidth=1.6)
    ax.fill_between(df.index, df[col], alpha=0.1, color=color)
    ax.set_ylabel(title, fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _, f=fmt: f.format(x=x)))

axes[0].set_title("Pakistan Remittance Predictors — All Signals (2010–2024)",
                  fontsize=13, fontweight="bold", pad=12)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
plt.tight_layout()
plt.savefig("outputs/02_all_features.png", bbox_inches="tight")
plt.close()
print("   Saved: outputs/02_all_features.png")

# ════════════════════════════════════════════════════════════════
# CHART 3 — Correlation Heatmap
# ════════════════════════════════════════════════════════════════
print("[3/5] Plotting correlation heatmap...")
corr_cols = ["remittances_usd_mn", "oil_price_brent",
             "usd_pkr", "gulf_employment_idx", "pol_stability_idx"]
corr_labels = ["Remittances", "Oil Price",
               "USD/PKR Rate", "Gulf Employment", "Political Stability"]

corr = df[corr_cols].corr()
corr.index   = corr_labels
corr.columns = corr_labels

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, annot_kws={"size": 11},
            cbar_kws={"shrink": 0.8})
ax.set_title("Feature Correlation Matrix", fontsize=13,
             fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("outputs/03_correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("   Saved: outputs/03_correlation_heatmap.png")
print("\n   KEY CORRELATIONS WITH REMITTANCES:")
for col, label in zip(corr_cols[1:], corr_labels[1:]):
    val = df[corr_cols].corr().loc["remittances_usd_mn", col]
    print(f"   {label:25s}: r = {val:+.3f}")

# ════════════════════════════════════════════════════════════════
# CHART 4 — Cross-Correlation: Oil Price Lag (THE KEY FINDING)
# ════════════════════════════════════════════════════════════════
print("\n[4/5] Plotting oil price lag cross-correlation...")

lags = range(0, 13)
ccf_values = []
for lag in lags:
    if lag == 0:
        corr_val = df["remittances_usd_mn"].corr(df["oil_price_brent"])
    else:
        corr_val = df["remittances_usd_mn"].corr(
            df["oil_price_brent"].shift(lag))
    ccf_values.append(corr_val)

fig, ax = plt.subplots(figsize=(10, 5))
colors = [RED if v == max(ccf_values) else BLUE for v in ccf_values]
bars = ax.bar(lags, ccf_values, color=colors, edgecolor="white",
              linewidth=0.5, alpha=0.85)

# Highlight peak lag
peak_lag = ccf_values.index(max(ccf_values))
ax.annotate(f"Peak at lag {peak_lag}\nr = {max(ccf_values):.3f}",
            xy=(peak_lag, max(ccf_values)),
            xytext=(peak_lag + 1.5, max(ccf_values) - 0.04),
            fontsize=10, color=RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Lag (months)", fontsize=11)
ax.set_ylabel("Correlation Coefficient (r)", fontsize=11)
ax.set_title("Cross-Correlation: Oil Price → Pakistan Remittances\n"
             "(How many months does oil price lead remittance changes?)",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xticks(list(lags))
ax.set_xticklabels([f"{l}m" for l in lags])
plt.tight_layout()
plt.savefig("outputs/04_oil_lag_crosscorr.png", bbox_inches="tight")
plt.close()
print(f"   Saved: outputs/04_oil_lag_crosscorr.png")
print(f"   *** KEY FINDING: Oil price strongest predictor at lag {peak_lag} months ***")
print(f"   *** This means oil prices predict remittances {peak_lag} months later ***")

# ════════════════════════════════════════════════════════════════
# CHART 5 — Seasonal Pattern by Month
# ════════════════════════════════════════════════════════════════
print("\n[5/5] Plotting seasonal patterns...")
monthly_avg = df.groupby("month")["remittances_usd_mn"].mean()
monthly_std = df.groupby("month")["remittances_usd_mn"].std()
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(range(1, 13), monthly_avg.values, color=BLUE,
       alpha=0.75, edgecolor="white")
ax.errorbar(range(1, 13), monthly_avg.values,
            yerr=monthly_std.values, fmt="none",
            color=RED, capsize=5, linewidth=2)

peak_month = monthly_avg.idxmax()
ax.annotate(f"Peak: {month_names[peak_month-1]}\n${monthly_avg[peak_month]:,.0f}M avg",
            xy=(peak_month, monthly_avg[peak_month]),
            xytext=(peak_month + 1.2, monthly_avg[peak_month] + 120),
            fontsize=9, color=RED, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=RED))

ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)
ax.set_ylabel("Average Remittances (USD Mn)", fontsize=11)
ax.set_title("Seasonal Pattern: Average Monthly Remittances (2010–2024)\n"
             "Error bars show standard deviation across years",
             fontsize=12, fontweight="bold", pad=12)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
plt.tight_layout()
plt.savefig("outputs/05_seasonal_pattern.png", bbox_inches="tight")
plt.close()
print("   Saved: outputs/05_seasonal_pattern.png")

print("\n✅ EDA Complete — 5 charts saved to outputs/")
print(f"   Peak remittance month: {month_names[monthly_avg.idxmax()-1]}")
print(f"   Lowest remittance month: {month_names[monthly_avg.idxmin()-1]}")
