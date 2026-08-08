"""Grid search over trail_pct × trail_activate_pct for G1/G2/L1/L2 splits.

Base: trial #1655 from gl_split_v1 study. Varies one split's trail params
while keeping the other 3 splits at base values. OOS only (Mar-Jul 2026).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import optuna
from concurrent.futures import ProcessPoolExecutor, as_completed
from scripts.analysis.split_backtest import _build_snapshot_for_trial, run_split_backtest
from test_full import load_all_picks
from optimize_combined import DATA_DIRS
import test_green_candle_combined as tgc

WORKERS = 6
TRAIL_VALS = [i * 0.5 for i in range(7)]  # 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0

SPLITS = [
    ("G1", "g1_trail_pct", "g1_trail_activate_pct"),
    ("G2", "g2_trail_pct", "g2_trail_activate_pct"),
    ("L1", "l1_trail_pct", "l1_trail_activate_pct"),
    ("L2", "l2_trail_pct", "l2_trail_activate_pct"),
]

_worker_cache = {}


def _run_combo(args):
    """Run a single (split_name, trail_key, act_key, trail_val, act_val, base_params, oos_dates) combo."""
    split_name, trail_key, act_key, trail_val, act_val, base_params, oos_dates = args
    try:
        params = dict(base_params)
        params[trail_key] = trail_val
        params[act_key] = act_val
        snapshot = _build_snapshot_for_trial(params)
        if "picks" not in _worker_cache:
            _, dp = load_all_picks(DATA_DIRS)
            _worker_cache["picks"] = dp
        dp = _worker_cache["picks"]
        result = run_split_backtest(dp, oos_dates, params_snapshot=snapshot)
        sp = result["splits"].get(split_name, {})
        return {
            "split": split_name,
            "trail": trail_val,
            "act": act_val,
            "pnl": result["total_pnl"],
            "wr": result["wr"],
            "pf": result["pf"],
            "dd": result["max_dd_pct"],
            "trades": result["n"],
            "sp_trades": sp.get("trades", 0),
            "sp_pnl": sp.get("pnl", 0),
            "sp_wr": sp.get("wr", 0),
            "sp_pf": sp.get("pf", 0),
        }
    except Exception as e:
        return {"split": split_name, "trail": trail_val, "act": act_val, "error": str(e)}


def main():
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True

    # Load study & base params
    storage = optuna.storages.RDBStorage(url="postgresql://postgres@127.0.0.1:5432/optuna_gl_split")
    study = optuna.load_study(study_name="gl_split_v1", storage=storage)
    base_params = dict(study.trials[1655].params)

    # Load data for date ranges
    print("Loading data...")
    all_dates, _ = load_all_picks(DATA_DIRS)
    oos_dates = sorted([d for d in all_dates if "2026-03-17" <= d <= "2026-07-10"])
    print(f"  OOS: {len(oos_dates)} days")

    # Build all tasks: 4 splits × 7×7 = 196 combos
    tasks = []
    for split_name, trail_key, act_key in SPLITS:
        for trail_val in TRAIL_VALS:
            for act_val in TRAIL_VALS:
                tasks.append((split_name, trail_key, act_key, trail_val, act_val, base_params, oos_dates))

    print(f"  Running {len(tasks)} combos with {WORKERS} workers...")

    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_run_combo, t): t for t in tasks}
        for future in as_completed(futures):
            done += 1
            r = future.result()
            if "error" in r:
                print(f"\n  ERROR {r['split']} trail={r['trail']} act={r['act']}: {r['error']}")
            else:
                results.append(r)
            if done % 20 == 0:
                print(f"\r  {done}/{len(tasks)} done...", end="", flush=True)
    print()

    # Print results per split
    for split_name, _, _ in SPLITS:
        print()
        print("=" * 100)
        print(f"  {split_name} — TOTAL PnL heatmap (trail_pct × trail_activate_pct)")
        print("=" * 100)

        # Filter to this split
        sr = [r for r in results if r["split"] == split_name]

        # Build lookup: (trail, act) -> split-specific values
        sp_pnl_map = {(r["trail"], r["act"]): r["sp_pnl"] for r in sr}
        sp_wr_map = {(r["trail"], r["act"]): r["sp_wr"] for r in sr}
        trades_map = {(r["trail"], r["act"]): r["sp_trades"] for r in sr}

        # Print PnL heatmap
        print(f"\n  {split_name} PnL ($):")
        header = f"  {'trail\\act':>10}"
        for act in TRAIL_VALS:
            header += f"  {act:>6.1f}%"
        print(header)
        print("  " + "-" * (10 + 9 * len(TRAIL_VALS)))
        for trail in TRAIL_VALS:
            row = f"  {trail:>9.1f}%"
            for act in TRAIL_VALS:
                pnl = sp_pnl_map.get((trail, act), 0)
                row += f"  {pnl/1000:>6.0f}k"
            print(row)

        # Print WR heatmap
        print(f"\n  {split_name} WR (%):")
        header = f"  {'trail\\act':>10}"
        for act in TRAIL_VALS:
            header += f"  {act:>6.1f}%"
        print(header)
        print("  " + "-" * (10 + 9 * len(TRAIL_VALS)))
        for trail in TRAIL_VALS:
            row = f"  {trail:>9.1f}%"
            for act in TRAIL_VALS:
                wr = sp_wr_map.get((trail, act), 0)
                row += f"  {wr:>6.1f}%"
            print(row)

        # Print Trades heatmap
        print(f"\n  {split_name} Trades:")
        header = f"  {'trail\\act':>10}"
        for act in TRAIL_VALS:
            header += f"  {act:>6.1f}%"
        print(header)
        print("  " + "-" * (10 + 9 * len(TRAIL_VALS)))
        for trail in TRAIL_VALS:
            row = f"  {trail:>9.1f}%"
            for act in TRAIL_VALS:
                tr = trades_map.get((trail, act), 0)
                row += f"  {tr:>6}"
            print(row)

        # Find best combo
        best = max(sr, key=lambda x: x["sp_pnl"])
        print(f"\n  BEST: trail={best['trail']:.1f}% act={best['act']:.1f}% → "
              f"PnL=${best['sp_pnl']:,.0f}  WR={best['sp_wr']:.1f}%  PF={best['sp_pf']:.2f}  Trades={best['sp_trades']}")

    # Print overall (all splits combined) best combos
    print()
    print("=" * 100)
    print("  OVERALL BEST COMBOS (all splits combined)")
    print("=" * 100)

    # Group by (trail, act) and sum PnL across splits
    from collections import defaultdict
    overall = defaultdict(lambda: {"pnl": 0, "wr": 0, "trades": 0})
    for r in results:
        key = (r["trail"], r["act"])
        overall[key]["pnl"] += r["sp_pnl"]
        overall[key]["trades"] += r["sp_trades"]

    ranked = sorted(overall.items(), key=lambda x: x[1]["pnl"], reverse=True)
    print(f"\n  Top 20 (trail_pct, act_pct) by combined PnL across G1+G2+L1+L2:")
    print(f"  {'Rank':<5} {'trail%':>7} {'act%':>7} {'Total PnL':>14} {'Trades':>8}")
    print(f"  {'-'*5} {'-'*7} {'-'*7} {'-'*14} {'-'*8}")
    for i, (key, data) in enumerate(ranked[:20], 1):
        print(f"  {i:<5} {key[0]:>7.1f} {key[1]:>7.1f} ${data['pnl']:>12,.0f} {data['trades']:>8}")

    print("\nDone.")


if __name__ == "__main__":
    main()
