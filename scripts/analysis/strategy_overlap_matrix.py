"""Systematic strategy overlap matrix on 2026.

For each strategy individually:
  - Enable ONLY that strategy with #254's tuned params (or baseline defaults)
  - Forward-test on 2026 (the test universe)
  - Collect (date, ticker) tuples where the strategy fired

Then build a 22×22 overlap matrix:
  overlap[i][j] = |trades(i) ∩ trades(j)| / |trades(i) ∪ trades(j)|  (Jaccard)

Identifies:
  - Which strategies fire on the SAME trade opportunities (redundant)
  - Which fire on DIFFERENT opportunities (complementary)
  - Which fire zero times (dead — R, X, etc.)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
SOURCE_CONFIG = "config/trial_254_w7_extracted.json"  # has many tuned params
DATA_DIRS_2026 = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]
OUTDIR = "results/strategy_overlap"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]


def forward_single_strategy(strat):
    """Forward-test 2026 with ONLY this strategy enabled."""
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD

    with open(SOURCE_CONFIG) as f: data = json.load(f)
    src_params = data.get("params", data)
    with open(BASELINE) as f: baseline = json.load(f)
    merged = dict(baseline); merged.update(src_params)
    # Force enable only this strategy
    for s in ALL_STRATS:
        merged[f"enable_{s}"] = (s == strat)
    # CRITICAL: must override X's kill switch since #254 has it at 9999
    if strat == "x":
        merged["x_min_first_leg_gain_pct"] = merged.get("x_min_first_leg_gain_pct", 5.0)
        if merged["x_min_first_leg_gain_pct"] >= 9999:
            merged["x_min_first_leg_gain_pct"] = 5.0

    set_strategy_params(merged)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.MARGIN_MULTIPLIER = 1.0

    dirs = [d for d in DATA_DIRS_2026 if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if d.startswith("2026")])

    cash = STARTING_CASH
    trade_keys = set()  # (date, ticker) tuples
    n_trades = 0
    total_pnl = 0
    for d in dates:
        day_picks = picks_by_date.get(d, [])
        if not day_picks: continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception:
            continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                trade_keys.add((d, st["ticker"]))
                n_trades += 1
                total_pnl += st["pnl"]
        cash = end_c + (unset if is_cash else 0)

    return {
        "strategy": strat,
        "n_trades": n_trades,
        "n_unique_keys": len(trade_keys),
        "total_pnl": float(total_pnl),
        "final_equity": float(cash),
        "trade_keys": list(trade_keys),
    }


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Forward-testing each of {len(ALL_STRATS)} strategies individually on 2026...\n")

    results = {}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(forward_single_strategy, s): s for s in ALL_STRATS}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                r = fut.result()
                results[s] = r
                print(f"  {s.upper():<2}: {r['n_trades']:>4} trades  ({r['n_unique_keys']} unique (date,ticker))  PnL ${r['total_pnl']:>+10,.0f}")
            except Exception as e:
                print(f"  {s.upper()}: FAILED — {e}")

    # Save raw
    with open(f"{OUTDIR}/individual_results.json", "w") as f:
        json.dump({s: {**r, "trade_keys": [list(k) for k in r["trade_keys"]]}
                   for s, r in results.items()}, f, indent=2)

    # Build overlap matrix (Jaccard similarity)
    print(f"\n=== Building overlap matrix ===")
    strats = sorted(results.keys())
    n = len(strats)
    overlap = np.zeros((n, n))
    co_count = np.zeros((n, n), dtype=int)

    for i, s_i in enumerate(strats):
        keys_i = set(tuple(k) for k in results[s_i]["trade_keys"])
        for j, s_j in enumerate(strats):
            keys_j = set(tuple(k) for k in results[s_j]["trade_keys"])
            if not keys_i and not keys_j:
                overlap[i, j] = 0
            else:
                union = keys_i | keys_j
                intersection = keys_i & keys_j
                overlap[i, j] = len(intersection) / len(union) if union else 0
                co_count[i, j] = len(intersection)

    # Print matrix
    print(f"\n  Jaccard overlap (1.0 = identical trades, 0.0 = no overlap):")
    print(f"  {'':<4}", end="")
    for s in strats: print(f"  {s.upper():<5}", end="")
    print()
    for i, s_i in enumerate(strats):
        print(f"  {s_i.upper():<4}", end="")
        for j in range(n):
            v = overlap[i, j]
            if i == j: cell = "  -   "
            elif v == 0: cell = "  ·   "
            elif v < 0.05: cell = f"  {v:.2f} "
            else: cell = f" {v:.2f} "
            print(cell, end="")
        print()

    # Most-overlapping pairs (excluding self)
    print(f"\n=== Most overlapping pairs (excluding self) ===")
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if overlap[i, j] > 0:
                pairs.append((strats[i], strats[j], overlap[i, j], co_count[i, j]))
    pairs.sort(key=lambda x: -x[2])
    for s_i, s_j, jacc, count in pairs[:15]:
        n_i = results[s_i]["n_unique_keys"]
        n_j = results[s_j]["n_unique_keys"]
        print(f"  {s_i.upper()}-{s_j.upper()}:  Jaccard={jacc:.3f}  shared {count} of (|{s_i.upper()}|={n_i}, |{s_j.upper()}|={n_j})")

    # Compute which strategies are "live" (fire >0 trades)
    live = [s for s in strats if results[s]["n_trades"] > 0]
    dead = [s for s in strats if results[s]["n_trades"] == 0]
    print(f"\n=== Status ===")
    print(f"  LIVE strategies ({len(live)}): {','.join(s.upper() for s in live)}")
    print(f"  DEAD (fire zero on 2026): {','.join(s.upper() for s in dead)}")

    # Save matrix
    np.save(f"{OUTDIR}/overlap_jaccard.npy", overlap)
    with open(f"{OUTDIR}/overlap_pairs.json", "w") as f:
        json.dump([{"a": p[0], "b": p[1], "jaccard": p[2], "shared": p[3]} for p in pairs],
                   f, indent=2)

    # Heatmap chart
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(overlap, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([s.upper() for s in strats])
    ax.set_yticklabels([s.upper() for s in strats])
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    # Annotate
    for i in range(n):
        for j in range(n):
            v = overlap[i, j]
            if i == j: continue
            if v > 0.5: ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                  color="white" if v > 0.7 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, label="Jaccard overlap")
    ax.set_title("Strategy overlap matrix (2026) — Jaccard similarity of (date, ticker) sets\n"
                  "Each strategy run alone with #254's params; G+L are the W10 alpha pair")
    fig.tight_layout()
    chart_path = f"{OUTDIR}/overlap_heatmap.png"
    fig.savefig(chart_path, dpi=140); plt.close(fig)
    print(f"\n  Wrote {chart_path}")
    print(f"  Wrote {OUTDIR}/overlap_pairs.json")
    print(f"  Wrote {OUTDIR}/individual_results.json")


if __name__ == "__main__":
    main()
