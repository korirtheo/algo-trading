"""Compare deployed config vs optimized trial #1655 variations.
Also does a $10k daily stress test for all 4 configs.
OOS only: Mar 17 - Jul 10 2026.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import optuna
import test_green_candle_combined as tgc
from optimize_combined import (
    run_combined_backtest, set_strategy_params, _build_param_snapshot,
    _param_lock, DATA_DIRS, ALL_STRATS
)
from optimize_gl_split import _map_split_to_standard
from test_full import load_all_picks

# ── 1) Build param snapshots for all 4 configs ──────────────────────────

def build_deployed_snapshot():
    """Load best trial from deployed oglhmafp_v5 study."""
    storage = optuna.storages.RDBStorage(url="postgresql://postgres@127.0.0.1:5432/optuna_oglhmafp")
    study = optuna.load_study(study_name="oglhmafp_v5", storage=storage)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    best = max(completed, key=lambda t: t.value)
    params = dict(best.params)
    print(f"  Deployed trial #{best.number} (score: {best.value:,.0f})")
    # Inject enable flags — deployed runs all strategies
    for s in ALL_STRATS:
        params[f"enable_{s}"] = True
    with _param_lock:
        set_strategy_params(params)
        tgc.USE_DYNAMIC_SLIPPAGE = True
        tgc.USE_MULTIWINDOW_SLIPPAGE = True
        snapshot = _build_param_snapshot()
    return snapshot


def build_split_trial_snapshot(study_name, db, trial_num, overrides=None):
    """Load a trial from a split study, map to standard params, apply overrides."""
    storage = optuna.storages.RDBStorage(url=f"postgresql://postgres@127.0.0.1:5432/{db}")
    study = optuna.load_study(study_name=study_name, storage=storage)
    trial = study.trials[trial_num]
    params = dict(trial.params)
    std = _map_split_to_standard(params)
    # Only G+L strategies
    for s in ALL_STRATS:
        std[f"enable_{s}"] = (s in ("g", "l"))
    if overrides:
        std.update(overrides)
    with _param_lock:
        set_strategy_params(std)
        tgc.USE_DYNAMIC_SLIPPAGE = True
        tgc.USE_MULTIWINDOW_SLIPPAGE = True
        snapshot = _build_param_snapshot()
    return snapshot


# ── 2) Stress test: $10k daily sizing ────────────────────────────────────

def run_stress_test(dp, dates, params_snapshot, label, daily_cap=10000):
    """Run backtest with $10k account — captures per-day PnL."""
    cash = float(daily_cap)
    unsettled = 0.0
    daily_equity = [float(daily_cap)]
    daily_pnls = []

    for d in dates:
        cash += unsettled
        unsettled = 0.0
        eq_start = cash

        picks = dp.get(d, [])
        cash_account = cash < 25000

        if tgc.NEWS_FILTER_ENABLED and (tgc.NEWS_MIN_ARTICLES > 0 or tgc.NEWS_REQUIRE_CATALYST):
            from news_filter import filter_picks as _news_filter
            picks = _news_filter(picks, d, min_articles=tgc.NEWS_MIN_ARTICLES, require_catalyst=tgc.NEWS_REQUIRE_CATALYST)

        if tgc.MIN_FAVORABILITY_THRESHOLD > 0 and tgc._MACRO_DATA:
            _score, _ = tgc.compute_day_favorability(d, picks)
            if _score is not None and _score < tgc.MIN_FAVORABILITY_THRESHOLD:
                daily_equity.append(cash + unsettled)
                daily_pnls.append((d, 0.0))
                continue

        states, cash, unsettled, _ = tgc.simulate_day_combined(
            picks, cash, cash_account, params=params_snapshot
        )

        eq_end = cash + unsettled
        daily_pnl = eq_end - eq_start
        daily_pnls.append((d, daily_pnl))
        daily_equity.append(eq_end)

    # Compute stats
    import numpy as np
    eq = np.array(daily_equity)
    pnl_arr = np.array([p for _, p in daily_pnls])

    green_days = int(np.sum(pnl_arr > 0))
    red_days = int(np.sum(pnl_arr < 0))
    flat_days = int(np.sum(pnl_arr == 0))
    total_days = len(pnl_arr)

    total_pnl = float(eq[-1] - eq[0])
    avg_daily_pnl = float(pnl_arr.mean()) if len(pnl_arr) > 0 else 0
    avg_daily_pct = (avg_daily_pnl / daily_cap) * 100 if daily_cap > 0 else 0

    # Max drawdown from equity curve
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    max_dd = float(dd.min())

    # Profit factor
    gross_win = float(pnl_arr[pnl_arr > 0].sum()) if np.any(pnl_arr > 0) else 0
    gross_loss = float(abs(pnl_arr[pnl_arr < 0].sum())) if np.any(pnl_arr < 0) else 1e-9
    pf = gross_win / gross_loss if gross_loss > 0 else 99.0

    wr = (green_days / total_days * 100) if total_days > 0 else 0

    # Avg win / avg loss
    avg_win = float(pnl_arr[pnl_arr > 0].mean()) if np.any(pnl_arr > 0) else 0
    avg_loss = float(pnl_arr[pnl_arr < 0].mean()) if np.any(pnl_arr < 0) else 0

    return {
        "label": label,
        "daily_cap": daily_cap,
        "total_pnl": total_pnl,
        "total_pnl_pct": (total_pnl / daily_cap) * 100,
        "green_days": green_days,
        "red_days": red_days,
        "flat_days": flat_days,
        "total_days": total_days,
        "win_rate": wr,
        "avg_daily_pnl": avg_daily_pnl,
        "avg_daily_pct": avg_daily_pct,
        "max_dd": max_dd,
        "pf": pf,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "daily_pnls": daily_pnls,
    }


# ── 3) Main ──────────────────────────────────────────────────────────────

def main():
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True

    # Load data
    print("Loading data...")
    all_dates, dp = load_all_picks(DATA_DIRS)
    oos_dates = sorted([d for d in all_dates if "2026-03-17" <= d <= "2026-07-10"])
    print(f"  OOS: {len(oos_dates)} days (Mar 17 - Jul 10 2026)\n")

    # Build all 4 configs
    configs = []

    print("Building config 1: Deployed oglhmafp_v5...")
    snap_deployed = build_deployed_snapshot()
    configs.append(("DEPLOYED", snap_deployed))

    print("\nBuilding config 2: Trial #1655 base (original params)...")
    snap_base = build_split_trial_snapshot("gl_split_v1", "optuna_gl_split", 1655)
    configs.append(("#1655 BASE", snap_base))

    print("\nBuilding config 3: Trial #1655 + G1 trail=0.5%/act=0.5%...")
    snap_g1 = build_split_trial_snapshot("gl_split_v1", "optuna_gl_split", 1655,
                                         overrides={"g_trail_pct": 0.5, "g_trail_activate_pct": 0.5})
    configs.append(("#1655+G1", snap_g1))

    print("\nBuilding config 4: Trial #1655 + G1 trail=0.5%/act=0.5% + L1 trail=0%/act=0.5%...")
    snap_both = build_split_trial_snapshot("gl_split_v1", "optuna_gl_split", 1655,
                                           overrides={"g_trail_pct": 0.5, "g_trail_activate_pct": 0.5,
                                                      "l_trail_pct": 0.0, "l_trail_activate_pct": 0.5})
    configs.append(("#1655+BOTH", snap_both))

    # ── Run full account backtests ──
    print("\n" + "=" * 100)
    print("  FULL ACCOUNT BACKTEST — $25K start, OOS Mar 17 - Jul 10 2026")
    print("=" * 100)

    full_results = []
    for label, snap in configs:
        print(f"\n  Running {label}...")
        result = run_combined_backtest(dp, oos_dates, params_snapshot=snap)
        full_results.append((label, result))

    # Print comparison table
    print("\n" + "=" * 100)
    print("  FULL ACCOUNT COMPARISON")
    print("=" * 100)
    print(f"  {'Config':<16} {'PnL':>12} {'WR%':>7} {'PF':>7} {'Trades':>7} {'MaxDD%':>8}")
    print(f"  {'-'*16} {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
    for label, r in full_results:
        n = r.get("n", 0)
        if n > 0:
            print(f"  {label:<16} ${r['total_pnl']:>10,.0f} {r.get('wr', 0):>6.1f}% {r.get('pf', 0):>6.2f} {n:>7} {r.get('max_dd_pct', 0):>7.1f}%")
        else:
            print(f"  {label:<16} {'No trades':>12}")

    # Per-strategy breakdown for full account
    print("\n" + "=" * 100)
    print("  PER-STRATEGY BREAKDOWN")
    print("=" * 100)
    for label, r in full_results:
        strats = r.get("strats", {})
        if not strats:
            continue
        print(f"\n  {label}:")
        print(f"  {'Strategy':<10} {'Trades':>7} {'WR%':>7} {'PnL':>12} {'AvgPnL':>10}")
        print(f"  {'-'*10} {'-'*7} {'-'*7} {'-'*12} {'-'*10}")
        for s in sorted(strats.keys()):
            sd = strats[s]
            sn = sd["n"]
            if sn > 0:
                sw = sd["wins"] / sn * 100
                sp = sd["pnl"]
                avg = sp / sn
                print(f"  {s:<10} {sn:>7} {sw:>6.1f}% ${sp:>10,.0f} ${avg:>8,.0f}")

    # ── Stress test: $10k daily ──
    print("\n\n" + "=" * 100)
    print("  STRESS TEST — $10k daily sizing, OOS Mar 17 - Jul 10 2026")
    print("=" * 100)

    stress_results = []
    for label, snap in configs:
        print(f"\n  Running {label}...")
        sr = run_stress_test(dp, oos_dates, snap, label, daily_cap=10000)
        stress_results.append(sr)

    # Print stress test comparison
    print("\n" + "=" * 100)
    print("  STRESS TEST COMPARISON")
    print("=" * 100)
    print(f"  {'Config':<16} {'PnL':>10} {'PnL%':>8} {'WinR%':>7} {'G':>4} {'R':>4} {'F':>4} {'MaxDD%':>8} {'PF':>7} {'AvgDay%':>9}")
    print(f"  {'-'*16} {'-'*10} {'-'*8} {'-'*7} {'-'*4} {'-'*4} {'-'*4} {'-'*8} {'-'*7} {'-'*9}")
    for sr in stress_results:
        print(f"  {sr['label']:<16} ${sr['total_pnl']:>8,.0f} {sr['total_pnl_pct']:>7.1f}% "
              f"{sr['win_rate']:>6.1f}% {sr['green_days']:>4} {sr['red_days']:>4} {sr['flat_days']:>4} "
              f"{sr['max_dd']:>7.1f}% {sr['pf']:>6.2f} {sr['avg_daily_pct']:>8.2f}%")

    # Best / worst / avg day details
    print("\n" + "=" * 100)
    print("  DAILY PnL DETAILS")
    print("=" * 100)
    for sr in stress_results:
        pnl_arr = [p for _, p in sr["daily_pnls"]]
        import numpy as np
        pa = np.array(pnl_arr)
        best_day = float(pa.max()) if len(pa) > 0 else 0
        worst_day = float(pa.min()) if len(pa) > 0 else 0
        median_day = float(np.median(pa)) if len(pa) > 0 else 0
        std_day = float(pa.std()) if len(pa) > 0 else 0

        print(f"\n  {sr['label']}:")
        print(f"    Best day:     ${best_day:>+8,.0f}  ({best_day/sr['daily_cap']*100:>+.2f}%)")
        print(f"    Worst day:    ${worst_day:>+8,.0f}  ({worst_day/sr['daily_cap']*100:>+.2f}%)")
        print(f"    Avg day:      ${sr['avg_daily_pnl']:>+8,.0f}  ({sr['avg_daily_pct']:>+.2f}%)")
        print(f"    Median day:   ${median_day:>+8,.0f}  ({median_day/sr['daily_cap']*100:>+.2f}%)")
        print(f"    Std dev:      ${std_day:>8,.0f}  ({std_day/sr['daily_cap']*100:>.2f}%)")

    # Show individual day PnL for the best config
    print("\n" + "=" * 100)
    print("  BEST CONFIG — DAILY PnL LOG")
    print("=" * 100)
    best_sr = max(stress_results, key=lambda x: x["total_pnl"])
    print(f"  {best_sr['label']}:")
    print(f"  {'Date':<14} {'PnL':>10} {'%':>8} {'Cumul':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*8} {'-'*10}")
    cumul = 0.0
    for d, pnl in best_sr["daily_pnls"]:
        cumul += pnl
        marker = "+" if pnl > 0 else ("-" if pnl < 0 else " ")
        print(f"  {d:<14} ${pnl:>+8,.0f} {pnl/best_sr['daily_cap']*100:>+7.2f}% ${cumul:>+8,.0f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
