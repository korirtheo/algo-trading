"""Monthly regime trend chart 2019-2026.

Granular view of how the universe's regime composition has shifted
month-over-month. Reveals seasonality, regime transitions, and the
2026 microcap-thin concentration trend.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import os
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    print(f"Loading from {len(present)} dirs...")
    all_dates, picks = load_all_picks(present)
    print(f"Loaded {len(all_dates)} total dates ({all_dates[0]} -> {all_dates[-1]})")

    # Month key -> Counter of shapes
    month_counts = defaultdict(Counter)
    month_totals = defaultdict(int)
    for d in all_dates:
        month_key = d[:7]   # YYYY-MM
        shape = _classify(picks.get(d, []))
        month_counts[month_key][shape] += 1
        month_totals[month_key] += 1

    months = sorted(month_counts.keys())
    month_dates = [datetime.strptime(m + "-15", "%Y-%m-%d") for m in months]
    print(f"Months: {months[0]} -> {months[-1]} ({len(months)} months)")

    REGIMES = [
        ("microcap-thin",  "#d62728", "Microcap-thin (leader <$1)"),
        ("thin-microcap",  "#ff7f0e", "Thin-microcap (leader $1-3)"),
        ("liquid-normal",  "#2ca02c", "Liquid-normal (leader $3+, vol)"),
        ("broad-squeeze",  "#9467bd", "Broad-squeeze (many >50% gappers)"),
    ]

    # --- Main monthly trend chart ---
    fig, ax = plt.subplots(figsize=(16, 8))
    for shape, color, label in REGIMES:
        ys = [month_counts[m].get(shape, 0) / max(month_totals[m], 1) * 100
              for m in months]
        ax.plot(month_dates, ys, marker=".", markersize=4,
                linewidth=1.4, color=color, label=label, alpha=0.65)

        # Smoothed 3-month rolling average
        import numpy as np
        ys_arr = np.array(ys, dtype=float)
        smoothed = np.convolve(ys_arr, np.ones(3)/3, mode="same")
        # Fix edges
        if len(ys_arr) >= 3:
            smoothed[0] = ys_arr[:2].mean()
            smoothed[-1] = ys_arr[-2:].mean()
        ax.plot(month_dates, smoothed, linewidth=2.6, color=color)

    # Microcap-pump combined dashed line
    pump_ys = [(month_counts[m].get("microcap-thin", 0) +
                month_counts[m].get("thin-microcap", 0)) / max(month_totals[m], 1) * 100
               for m in months]
    import numpy as np
    pump_smooth = np.convolve(np.array(pump_ys, dtype=float), np.ones(3)/3, mode="same")
    pump_smooth[0] = pump_ys[0]; pump_smooth[-1] = pump_ys[-1]
    ax.plot(month_dates, pump_ys, color="black", alpha=0.3, linewidth=1.0)
    ax.plot(month_dates, pump_smooth, color="black", linewidth=3.0, linestyle="--",
            label="MICROCAP-PUMP combined (#124's regime, 3mo avg)")

    # Year separators
    for year in range(2020, 2027):
        ax.axvline(datetime(year, 1, 1), color="gray", alpha=0.15, linestyle=":")
        ax.text(datetime(year, 1, 1), 92, str(year), fontsize=9, alpha=0.5, ha="left")

    ax.set_title("Universe regime composition — monthly 2019-2026\n"
                 "Thin lines = monthly raw, thick = 3-month rolling average",
                 fontsize=13)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("% of trading days in month", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    fig.tight_layout()
    p = os.path.join(OUT_DIR, "regime_trends_monthly.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Wrote {p}")
    return p


if __name__ == "__main__":
    main()
