"""Quantify the monthly regime trends — linear fit + seasonality.

Reports:
  - Linear trend slope (% points per year) for each regime
  - R-squared (how reliable the trend is)
  - Seasonal averages (which calendar month favors which regime)

Outputs two charts:
  1. Monthly time series + linear trend lines overlaid
  2. Seasonal heatmap (regime % by calendar month, averaged across years)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import os
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime

from test_full import load_all_picks
from scripts.analysis.squeeze_taxonomy_2021_2026 import day_signature, classify_shape
from strategies.regime_gate import classify_regime, compute_features

OUT_DIR = "results/regime_trends"

ALL_DIRS = [
    "stored_data_2019", "stored_data_2020", "stored_data_2021",
    "stored_data_2022", "stored_data_2023",
    "stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
    "stored_data_jul_sep_2024", "stored_data_oct_dec_2024",
    "stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
    "stored_data_jul_2025", "stored_data_oos",
    "stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026",
]


def _classify(picks):
    try:
        feats = compute_features(picks)
        regime = classify_regime(feats)
    except Exception:
        regime = "NORMAL"
    sig = day_signature(picks)
    if sig is None: return "empty"
    return classify_shape(sig, regime)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    present = [d for d in ALL_DIRS if os.path.exists(d)]
    all_dates, picks = load_all_picks(present)
    print(f"Loaded {len(all_dates)} dates ({all_dates[0]} -> {all_dates[-1]})")

    month_counts = defaultdict(Counter)
    month_totals = defaultdict(int)
    for d in all_dates:
        m = d[:7]
        month_counts[m][_classify(picks.get(d, []))] += 1
        month_totals[m] += 1

    months = sorted(month_counts.keys())
    month_dates = [datetime.strptime(m + "-15", "%Y-%m-%d") for m in months]

    # x-axis as years (float) for linear regression
    x_years = np.array([(d.year + (d.month - 1) / 12) for d in month_dates])

    REGIMES = [
        ("microcap-thin",  "#d62728", "Microcap-thin"),
        ("thin-microcap",  "#ff7f0e", "Thin-microcap"),
        ("liquid-normal",  "#2ca02c", "Liquid-normal"),
        ("broad-squeeze",  "#9467bd", "Broad-squeeze"),
    ]

    # ---- CHART 1: monthly + linear trend lines ----
    fig, ax = plt.subplots(figsize=(16, 8))

    print(f"\n{'='*68}")
    print(f"LINEAR TREND FITS (% points per year)")
    print(f"{'='*68}")
    print(f"  {'regime':<20} {'slope/yr':>10} {'intercept':>12} {'R^2':>8}")

    trend_summary = []
    for shape, color, label in REGIMES:
        ys = np.array([month_counts[m].get(shape, 0) / max(month_totals[m], 1) * 100
                       for m in months])
        slope, intercept = np.polyfit(x_years, ys, 1)
        y_pred = slope * x_years + intercept
        ss_res = np.sum((ys - y_pred) ** 2)
        ss_tot = np.sum((ys - ys.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Raw monthly (thin)
        ax.plot(month_dates, ys, marker=".", markersize=3,
                linewidth=1.0, color=color, alpha=0.4)
        # Linear trend (thick)
        ax.plot(month_dates, y_pred, linewidth=3.0, color=color,
                label=f"{label}: slope {slope:+.2f}pp/yr (R²={r2:.2f})")

        print(f"  {shape:<20} {slope:>+9.2f}  {intercept:>+11.2f}  {r2:>7.3f}")
        trend_summary.append((shape, color, label, slope, intercept, r2, ys))

    # Microcap-pump combined
    pump_ys = np.array([(month_counts[m].get("microcap-thin", 0) +
                          month_counts[m].get("thin-microcap", 0)) / max(month_totals[m], 1) * 100
                         for m in months])
    slope_p, intercept_p = np.polyfit(x_years, pump_ys, 1)
    pump_pred = slope_p * x_years + intercept_p
    r2_p = 1 - np.sum((pump_ys - pump_pred) ** 2) / np.sum((pump_ys - pump_ys.mean()) ** 2)
    ax.plot(month_dates, pump_ys, color="black", alpha=0.25, linewidth=1.0)
    ax.plot(month_dates, pump_pred, color="black", linewidth=3.5, linestyle="--",
            label=f"MICROCAP-PUMP combined: slope {slope_p:+.2f}pp/yr (R²={r2_p:.2f})")
    print(f"  {'microcap-pump':<20} {slope_p:>+9.2f}  {intercept_p:>+11.2f}  {r2_p:>7.3f}")

    for year in range(2020, 2027):
        ax.axvline(datetime(year, 1, 1), color="gray", alpha=0.15, linestyle=":")

    ax.set_title("Monthly regime trends 2019-2026 with linear fits\n"
                 "Slope = percentage-point change per year",
                 fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("% of trading days in month")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "regime_monthly_with_trend.png")
    fig.savefig(p1, dpi=150); plt.close(fig)
    print(f"\nWrote {p1}")

    # ---- CHART 2: seasonality (avg % by calendar month, across years) ----
    print(f"\n{'='*68}")
    print(f"SEASONALITY (avg % across years 2019-2026, by calendar month)")
    print(f"{'='*68}")
    print(f"  {'month':<6} " + " ".join(f"{s:>14}" for s, _, _ in REGIMES))

    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    seasonal = {}
    for mi in range(1, 13):
        seasonal[mi] = {}
        for shape, _, _ in REGIMES:
            vals = []
            for m in months:
                if int(m[5:7]) == mi:
                    pct = month_counts[m].get(shape, 0) / max(month_totals[m], 1) * 100
                    vals.append(pct)
            seasonal[mi][shape] = np.mean(vals) if vals else 0

    for mi in range(1, 13):
        row = f"  {month_names[mi-1]:<6} "
        for shape, _, _ in REGIMES:
            row += f"{seasonal[mi][shape]:>13.1f}%"
        print(row)

    # Plot seasonality
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(1, 13)
    width = 0.2
    for i, (shape, color, label) in enumerate(REGIMES):
        ys = [seasonal[mi][shape] for mi in range(1, 13)]
        ax.bar(x + (i - 1.5) * width, ys, width, color=color, label=label, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(month_names)
    ax.set_title("Seasonal regime composition (avg % by calendar month across 2019-2026)", fontsize=12)
    ax.set_xlabel("Calendar month")
    ax.set_ylabel("Avg % of days in that month")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "regime_seasonality.png")
    fig.savefig(p2, dpi=150); plt.close(fig)
    print(f"\nWrote {p2}")


if __name__ == "__main__":
    main()
