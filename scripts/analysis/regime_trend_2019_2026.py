"""Plot regime frequency trends 2019-2026.

For each year, classify each trading day's shape (microcap-thin,
thin-microcap, liquid-normal, broad-squeeze, mega-cap, corp-action,
empty/dead) using the same classifier we used for the W3/W4 shape filter.

Output: line chart showing how the universe's regime composition has
shifted year-over-year. This tells us whether the microcap-pump regime
is in structural decline (which would explain #124's 2026 weakness) or
just a noisy bad year.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import os
from collections import Counter, defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from test_full import load_all_picks
from scripts.analysis.squeeze_taxonomy_2021_2026 import day_signature, classify_shape
from strategies.regime_gate import classify_regime, compute_features

OUT_DIR = "results/regime_trends"

# Year -> list of data directories
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

SHAPE_COLORS = {
    "microcap-thin":  "#d62728",  # red
    "thin-microcap":  "#ff7f0e",  # orange
    "liquid-normal":  "#2ca02c",  # green
    "broad-squeeze":  "#9467bd",  # purple
    "mega-cap":       "#1f77b4",  # blue
    "corp-action":    "#8c564b",  # brown
    "empty":          "#bbbbbb",  # gray
    "dead":           "#444444",  # dark gray
}


def _classify(picks):
    try:
        feats = compute_features(picks)
        regime = classify_regime(feats)
    except Exception:
        regime = "NORMAL"
    sig = day_signature(picks)
    if sig is None:
        return "empty"
    return classify_shape(sig, regime)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # year -> shape -> count
    year_shape_counts = {}
    year_totals = {}

    for year, dirs in YEAR_DIRS.items():
        present = [d for d in dirs if os.path.exists(d)]
        if not present:
            print(f"[skip] {year}: no data dirs"); continue
        all_dates, picks = load_all_picks(present)
        year_dates = [d for d in all_dates if d.startswith(year)]
        if not year_dates:
            print(f"[skip] {year}: no dates"); continue

        counts = Counter()
        for d in year_dates:
            shape = _classify(picks.get(d, []))
            counts[shape] += 1

        year_shape_counts[year] = counts
        year_totals[year] = len(year_dates)
        print(f"{year}: {len(year_dates)} days  {dict(counts)}")

    if not year_shape_counts:
        print("No data."); return

    years_sorted = sorted(year_shape_counts.keys())
    all_shapes = sorted(set(s for c in year_shape_counts.values() for s in c.keys()))

    # --- Chart 1: Absolute counts per year ---
    fig, ax = plt.subplots(figsize=(13, 7))
    for shape in all_shapes:
        ys = [year_shape_counts[y].get(shape, 0) for y in years_sorted]
        if max(ys) == 0:
            continue
        ax.plot(years_sorted, ys, marker="o", linewidth=2.0,
                color=SHAPE_COLORS.get(shape, "#888"),
                label=f"{shape} (peak={max(ys)})")
    ax.set_title("Universe regime composition — days per shape per year (2019-2026)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of trading days")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "regime_days_absolute.png")
    fig.savefig(p1, dpi=140); plt.close(fig)
    print(f"\nWrote {p1}")

    # --- Chart 2: Percentage share per year ---
    fig, ax = plt.subplots(figsize=(13, 7))
    for shape in all_shapes:
        ys = [year_shape_counts[y].get(shape, 0) / year_totals[y] * 100 for y in years_sorted]
        if max(ys) == 0:
            continue
        ax.plot(years_sorted, ys, marker="o", linewidth=2.0,
                color=SHAPE_COLORS.get(shape, "#888"),
                label=f"{shape}")
    ax.set_title("Universe regime composition — % share of trading days per year (2019-2026)")
    ax.set_xlabel("Year")
    ax.set_ylabel("% of trading days")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    ax.set_ylim(0, max(100, ax.get_ylim()[1]))
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "regime_days_pct.png")
    fig.savefig(p2, dpi=140); plt.close(fig)
    print(f"Wrote {p2}")

    # --- Chart 3: Microcap-pump (microcap-thin + thin-microcap) trend specifically ---
    fig, ax = plt.subplots(figsize=(13, 7))
    pump_pct = []
    pump_counts = []
    for y in years_sorted:
        n_pump = year_shape_counts[y].get("microcap-thin", 0) + year_shape_counts[y].get("thin-microcap", 0)
        pump_counts.append(n_pump)
        pump_pct.append(n_pump / year_totals[y] * 100)
    ax2 = ax.twinx()
    ax.bar(years_sorted, pump_counts, alpha=0.3, color="#d62728", label="Microcap-pump days (count, left)")
    ax2.plot(years_sorted, pump_pct, marker="o", linewidth=3.0, color="#d62728",
             label="Microcap-pump days (% of year, right)")
    ax.set_title("MICROCAP-PUMP regime (#124's training universe) — 2019-2026 trend")
    ax.set_xlabel("Year")
    ax.set_ylabel("Days (count)")
    ax2.set_ylabel("% of trading days")
    ax.grid(True, alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=10)
    fig.tight_layout()
    p3 = os.path.join(OUT_DIR, "microcap_pump_trend.png")
    fig.savefig(p3, dpi=140); plt.close(fig)
    print(f"Wrote {p3}")

    # --- Text summary ---
    print(f"\n{'='*86}")
    print(f"REGIME TREND TABLE — days per shape per year")
    print(f"{'='*86}")
    print(f"  {'shape':<16} " + " ".join(f"{y:>6}" for y in years_sorted) + f"  {'total':>7}")
    for shape in all_shapes:
        row = f"  {shape:<16} "
        total = 0
        for y in years_sorted:
            n = year_shape_counts[y].get(shape, 0)
            row += f"{n:>6} "
            total += n
        row += f" {total:>7}"
        print(row)
    print(f"  {'TOTAL':<16} " + " ".join(f"{year_totals[y]:>6}" for y in years_sorted))

    print(f"\n{'='*86}")
    print(f"MICROCAP-PUMP CONCENTRATION — what fraction of each year was #124's regime?")
    print(f"{'='*86}")
    for y in years_sorted:
        n_pump = year_shape_counts[y].get("microcap-thin", 0) + year_shape_counts[y].get("thin-microcap", 0)
        pct = n_pump / year_totals[y] * 100
        print(f"  {y}: {n_pump:>3} / {year_totals[y]:>3} days = {pct:>5.1f}%")


if __name__ == "__main__":
    main()
