"""Focused regime trend chart 2019-2026 — only the 4 regimes we've discussed."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import os
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from test_full import load_all_picks
from scripts.analysis.squeeze_taxonomy_2021_2026 import day_signature, classify_shape
from strategies.regime_gate import classify_regime, compute_features

OUT_DIR = "results/regime_trends"

YEAR_DIRS = {
    "2019": ["stored_data_2019"],
    "2020": ["stored_data_2020"],
    "2021": ["stored_data_2021"],
    "2022": ["stored_data_2022"],
    "2023": ["stored_data_2023"],
    "2024": ["stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
              "stored_data_jul_sep_2024", "stored_data_oct_dec_2024"],
    "2025": ["stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
              "stored_data_jul_2025", "stored_data_oos"],
    "2026": ["stored_data", "stored_data_mar_may_2026"],
}


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
    year_shape_counts = {}
    year_totals = {}
    for year, dirs in YEAR_DIRS.items():
        present = [d for d in dirs if os.path.exists(d)]
        if not present: continue
        all_dates, picks = load_all_picks(present)
        ydates = [d for d in all_dates if d.startswith(year)]
        if not ydates: continue
        counts = Counter()
        for d in ydates:
            counts[_classify(picks.get(d, []))] += 1
        year_shape_counts[year] = counts
        year_totals[year] = len(ydates)

    years_sorted = sorted(year_shape_counts.keys())

    # The 4 regimes we've actually discussed
    REGIMES = [
        ("microcap-thin",  "#d62728", "Microcap-thin (leader <$1)"),
        ("thin-microcap",  "#ff7f0e", "Thin-microcap (leader $1-3)"),
        ("liquid-normal",  "#2ca02c", "Liquid-normal (leader $3+, decent vol)"),
        ("broad-squeeze",  "#9467bd", "Broad-squeeze (many >50% gappers)"),
    ]

    # --- Chart: % share, large and clean ---
    fig, ax = plt.subplots(figsize=(14, 8))
    for shape, color, label in REGIMES:
        ys = [year_shape_counts[y].get(shape, 0) / year_totals[y] * 100 for y in years_sorted]
        ax.plot(years_sorted, ys, marker="o", markersize=9,
                linewidth=2.8, color=color, label=label)
        # Annotate the latest point
        ax.annotate(f"{ys[-1]:.1f}%",
                    xy=(years_sorted[-1], ys[-1]),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=10, color=color, fontweight="bold",
                    va="center")

    # Add microcap-pump combined (microcap-thin + thin-microcap) — the regime #124 was trained on
    pump = [(year_shape_counts[y].get("microcap-thin", 0) +
             year_shape_counts[y].get("thin-microcap", 0)) / year_totals[y] * 100
            for y in years_sorted]
    ax.plot(years_sorted, pump, marker="s", markersize=10,
            linewidth=3.5, color="#000000", linestyle="--",
            label="MICROCAP-PUMP combined (#124's training regime)")
    ax.annotate(f"{pump[-1]:.1f}%",
                xy=(years_sorted[-1], pump[-1]),
                xytext=(8, 0), textcoords="offset points",
                fontsize=11, color="black", fontweight="bold", va="center")

    ax.set_title("Universe regime composition by year (2019-2026)\n"
                 "Microcap-pump is at record highs in 2026 — but #124's edge on it has decayed",
                 fontsize=13)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("% of trading days", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
    ax.set_ylim(0, 80)

    # Highlight 2026 as partial
    n_2026 = year_totals.get("2026", 0)
    ax.axvspan(6.5, 7.5, alpha=0.1, color="orange")
    ax.text(7, 75, f"2026 partial\n({n_2026} days)", ha="center", fontsize=9, alpha=0.7)

    fig.tight_layout()
    p = os.path.join(OUT_DIR, "regime_trends_clean.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Wrote {p}")
    return p


if __name__ == "__main__":
    main()
