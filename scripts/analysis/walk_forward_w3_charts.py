"""Visualize the W3 walk-forward result (train 2021-23, test 2024).

Loads the saved forward.json and produces:
  - results/walk_forward/W3_equity_2024.png        (equity curve)
  - results/walk_forward/W3_perday_2024.png        (per-day PnL bars)
  - results/walk_forward/W3_pnl_histogram_2024.png (distribution of daily PnL)
  - Top 10 winning + losing days printed to stdout
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FWD = "results/walk_forward/W3_train_2021_2022_2023_test_2024_forward.json"

def main():
    with open(FWD) as f:
        d = json.load(f)

    daily = d["daily"]
    dates = [r["date"] for r in daily]
    pnls = np.array([r["pnl"] for r in daily])
    eq = np.array(d["equity_curve"])
    n = len(daily)
    print(f"W3 (train 2021-23, test 2024) — {n} days")
    print(f"  Start ${eq[0]:,.0f} -> End ${eq[-1]:,.0f}  (PnL ${d['total_pnl']:+,.0f})")
    print(f"  Sharpe {d['sharpe']:.2f}  wins {d['wins']}/{n}  losses {d['losses']}/{n}")
    print(f"  Median day-PnL ${np.median(pnls):+,.0f}  mean ${pnls.mean():+,.0f}  std ${pnls.std():,.0f}")

    # Equity curve
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(range(len(eq)), eq, color="#d62728", linewidth=1.8)
    ax.axhline(eq[0], color="#aaa", linestyle="--", linewidth=0.8, label=f"start ${eq[0]:,.0f}")
    ax.set_yscale("log")
    ax.set_title("W3 forward equity (train 2021-23 -> test 2024)")
    ax.set_xlabel("Trading day index")
    ax.set_ylabel("Equity ($, log scale)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    fig.tight_layout()
    p1 = "results/walk_forward/W3_equity_2024.png"
    fig.savefig(p1, dpi=140); plt.close(fig)
    print(f"  Wrote {p1}")

    # Per-day PnL bars (sign-colored)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#2ca02c" if p > 0 else ("#d62728" if p < 0 else "#aaa") for p in pnls]
    ax.bar(range(n), pnls, color=colors, alpha=0.85)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_title(f"W3 forward 2024 — per-day PnL ({n} days)")
    ax.set_xlabel("Trading day index")
    ax.set_ylabel("Daily PnL ($)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    fig.tight_layout()
    p2 = "results/walk_forward/W3_perday_2024.png"
    fig.savefig(p2, dpi=140); plt.close(fig)
    print(f"  Wrote {p2}")

    # Histogram of pnls (signed-log scale for visibility)
    fig, ax = plt.subplots(figsize=(11, 5))
    nonzero = pnls[pnls != 0]
    if len(nonzero):
        ax.hist(nonzero, bins=60, color="#1f77b4", alpha=0.8, edgecolor="#fff")
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_title(f"W3 forward 2024 — daily PnL distribution ({len(nonzero)} non-zero days)")
    ax.set_xlabel("Daily PnL ($)")
    ax.set_ylabel("Count")
    ax.set_xscale("symlog")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p3 = "results/walk_forward/W3_pnl_histogram_2024.png"
    fig.savefig(p3, dpi=140); plt.close(fig)
    print(f"  Wrote {p3}")

    # Top + bottom days
    rows = list(zip(dates, pnls, eq[1:]))
    rows_sorted = sorted(rows, key=lambda r: r[1], reverse=True)
    print(f"\nTop 10 WINNING days (% of total PnL):")
    total = pnls.sum()
    for d_, p, e in rows_sorted[:10]:
        share = 100 * p / total if total else 0
        print(f"  {d_}  ${p:>+15,.0f}  share {share:>5.1f}%  end-equity ${e:>12,.0f}")
    print(f"\nTop 10 LOSING days:")
    for d_, p, e in rows_sorted[-10:]:
        share = 100 * p / total if total else 0
        print(f"  {d_}  ${p:>+15,.0f}  share {share:>5.1f}%  end-equity ${e:>12,.0f}")

    # PnL concentration
    cum_share = np.cumsum(sorted(pnls, reverse=True)) / total * 100
    for k in [3, 5, 10, 20]:
        if k <= n:
            print(f"  Top {k} days' share of total PnL: {cum_share[k-1]:.1f}%")


if __name__ == "__main__":
    main()
