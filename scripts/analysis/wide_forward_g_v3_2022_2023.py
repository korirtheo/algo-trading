"""
Wide forward test: best 50 + random 50 G+V3 trials on 2022-2023 OOS.
Queries Optuna study, runs each trial's params OOS, reports correlation/win-rate.
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import optuna
import test_green_candle_combined as tgc
from test_full import load_all_picks
from optimize_combined import set_strategy_params as _oc_set_params
from optimize_combined import _build_param_snapshot, _param_lock

STARTING_CASH = 25_000
POSITION_PCT = 0.30
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
BASELINE_PATH = "config/trial_g511_l626_v3_overlay.json"

# OOS: only 2022-2023 (never used in training)
DATA_DIRS = ["stored_data_2022", "stored_data_2023"]
DATE_LO = "2022-01-01"
DATE_HI = "2023-12-31"

DB_URL = "postgresql://postgres@127.0.0.1:5432/optuna_g_v3"


def _disable_adaptive_controls():
    tgc.MIN_PRICE = 0.0
    tgc.MAX_MODELED_SLIP_BP = 0.0
    tgc.MAX_CUM_DVOL_AT_ENTRY_M = 0.0
    tgc.MIN_ATR_PCT = 0.0
    tgc.MIN_FAVORABILITY_THRESHOLD = 0.0
    for s in ALL_STRATS:
        setattr(tgc, f"{s.upper()}_PARTICIPATION_CAP", 0.0)
    tgc.NEWS_FILTER_ENABLED = False


def build_picks(picks_by_date, dates):
    """Convert picks_by_date into lists for the period."""
    return {d: picks_by_date.get(d, []) for d in dates}


def run_g_backtest(picks_by_date, dates, g_params):
    """Run G-only backtest with given params on given dates."""
    with open(BASELINE_PATH) as f:
        baseline_full = json.load(f)
    p = dict(baseline_full["params"])
    for s in ALL_STRATS:
        p[f"enable_{s}"] = (s == "g")
    p["enable_x"] = False
    p.update(g_params)
    with _param_lock:
        _oc_set_params(p)
        _disable_adaptive_controls()
        snapshot = _build_param_snapshot()

    cash = float(STARTING_CASH)
    g_pnls = []
    for d in dates:
        dp = picks_by_date.get(d, [])
        if not dp:
            continue
        is_cash = cash < 25000
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                dp, cash, cash_account=is_cash, params=snapshot
            )
        except Exception:
            continue
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0:
                pnl = st.get("pnl", 0) or 0
                g_pnls.append(pnl)
        cash = end_c + (unset if is_cash else 0)

    n = len(g_pnls)
    if n == 0:
        return -9999, 0, 0, 0

    total = sum(g_pnls)
    wins = [p for p in g_pnls if p > 0]
    losses = [p for p in g_pnls if p <= 0]
    gw = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 1e-9
    pf = gw / gl
    wr = len(wins) / n * 100 if n > 0 else 0
    return total, pf, wr, n


def compute_v3_candidates(picks_by_date, dates, g_holds):
    """Compute V3 trade candidates."""
    candidates = []
    for d in dates:
        for p in picks_by_date.get(d, []):
            mh = p.get("market_hour_candles")
            if mh is None or len(mh) < 2:
                continue
            day_open = float(mh.iloc[0]["Open"])
            bar0_red = float(mh.iloc[0]["Close"]) <= day_open
            holds = g_holds.get((p["ticker"], d), [])

            scan_start = 1
            if not bar0_red and holds:
                g_exit_ts = max(x for _, x in holds)
                new_start = None
                for i in range(1, len(mh)):
                    if mh.index[i] > g_exit_ts:
                        new_start = i
                        break
                if new_start is None:
                    continue
                scan_start = new_start

            entry_idx = entry_price = entry_ts = None
            for i in range(scan_start, len(mh)):
                c = float(mh.iloc[i]["Close"])
                if c > day_open:
                    entry_idx, entry_price, entry_ts = i, c, mh.index[i]
                    break
            if entry_idx is None or entry_price is None or entry_price <= 0:
                continue
            bars_after = mh.iloc[entry_idx + 1:]
            if len(bars_after) == 0:
                continue
            candidates.append({
                "ticker": p["ticker"], "date": d,
                "entry_ts": entry_ts, "entry_price": entry_price,
                "mh": mh, "bars_after": bars_after,
            })
    return candidates


def _apply_caps(mh, ts, fill_price, requested):
    pre = mh.loc[mh.index <= ts]
    vol_shares = float(pre["Volume"].sum()) if len(pre) > 0 else 0
    dollar_vol = fill_price * vol_shares
    if dollar_vol <= 0:
        return 0, 0
    vol_limit = dollar_vol * (tgc.VOL_CAP_PCT / 100)
    v_eff_adj, _, _, v_regime = tgc._multi_window_effective_volume(mh, ts, fill_price)
    if tgc.MAX_REGIME_PARTICIPATION > 0 and v_regime > 0:
        vol_limit = min(vol_limit, v_regime * tgc.MAX_REGIME_PARTICIPATION)
    if tgc.MAX_2MIN_PARTICIPATION > 0 and v_eff_adj > 0:
        vol_limit = min(vol_limit, v_eff_adj * tgc.MAX_2MIN_PARTICIPATION)
    return min(requested, vol_limit), v_eff_adj


def _v3_sim(v3_params, candidate):
    entry_ts = candidate["entry_ts"]
    fill_price = candidate["entry_price"]
    mh = candidate["mh"]
    bars_after = candidate["bars_after"]
    pos_dollars = STARTING_CASH * POSITION_PCT
    capped_size, v_eff_adj = _apply_caps(mh, entry_ts, fill_price, pos_dollars)
    if capped_size < 50:
        return 0.0
    slip_in = tgc._entry_slip_pct(fill_price, capped_size, v_eff_adj)
    actual_entry = fill_price * (1 + slip_in / 100)
    shares = capped_size / actual_entry

    target = actual_entry * (1 + v3_params["target_pct"] / 100)
    stop = actual_entry * (1 - v3_params["stop_pct"] / 100)
    peak = actual_entry
    trail_stop = None
    max_bars = max(1, v3_params["time_limit_min"] // 2)

    exit_price = exit_ts = None
    for i, (ts, row) in enumerate(bars_after.iterrows()):
        if i >= max_bars:
            exit_price = float(row["Close"]); exit_ts = ts; break
        c_high = float(row["High"]); c_low = float(row["Low"])
        if c_high >= target:
            exit_price = target; exit_ts = ts; break
        if c_low <= stop:
            exit_price = stop; exit_ts = ts; break
        if c_high > peak:
            peak = c_high
        if ((peak / actual_entry - 1) * 100) >= v3_params["trail_activate_pct"]:
            nt = peak * (1 - v3_params["trail_pct"] / 100)
            if trail_stop is None or nt > trail_stop:
                trail_stop = nt
        if trail_stop is not None and c_low <= trail_stop:
            exit_price = trail_stop; exit_ts = ts; break

    if exit_price is None and len(bars_after) > 0:
        exit_price = float(bars_after.iloc[-1]["Close"])
        exit_ts = bars_after.index[-1]
    if exit_price is None or exit_ts is None:
        return 0.0

    fake_st = {"mh": mh}
    slip_out = tgc._exit_slip_pct(exit_price, shares, fake_st, exit_ts)
    actual_exit = exit_price * (1 - slip_out / 100)
    return shares * (actual_exit - actual_entry)


def compute_v3_pnl(v3_params, v3_candidates):
    total = 0.0
    n = 0
    gw, gl = 0.0, 0.0
    for c in v3_candidates:
        pnl = _v3_sim(v3_params, c)
        if pnl == 0.0:
            continue
        total += pnl
        n += 1
        if pnl > 0:
            gw += pnl
        else:
            gl += -pnl
    pf = gw / gl if gl > 0 else 0.0
    return total, pf, n, gw, gl


def run_oos_trial(params, v3_candidates, picks_by_date, dates):
    """Run one trial OOS. Extract g_params and v3_params from flat params dict."""
    g_params = {k: v for k, v in params.items() if k.startswith("g_")}
    v3_raw = {k.replace("v3_", ""): v for k, v in params.items() if k.startswith("v3_")}

    # G backtest
    g_pnl, g_pf, g_wr, g_n = run_g_backtest(picks_by_date, dates, g_params)

    # V3 overlay
    v3_pnl, v3_pf, v3_n, _, _ = compute_v3_pnl(v3_raw, v3_candidates)

    combined_pnl = g_pnl + v3_pnl
    return {
        "g_pnl": g_pnl, "g_pf": g_pf, "g_wr": g_wr, "g_n": g_n,
        "v3_pnl": v3_pnl, "v3_pf": v3_pf, "v3_n": v3_n,
        "combined_pnl": combined_pnl,
    }


def main():
    parser = argparse.ArgumentParser(description="Wide forward test G+V3 on 2022-2023")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--random-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db", default=DB_URL)
    parser.add_argument("--study", default="g_v3_joint")
    parser.add_argument("--out", default="results/wide_forward_g_v3_2022_2023.json")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load study
    storage = optuna.storages.RDBStorage(url=args.db)
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study,
        storage=storage,
        load_if_exists=True,
    )
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trials.sort(key=lambda t: t.value or -9e12, reverse=True)
    print(f"\nStudy: {len(trials)} completed trials", flush=True)

    # Select top N + random N
    top_set = trials[:args.top_n]
    remainder = trials[args.top_n:]
    if len(remainder) < args.random_n:
        print(f"WARNING: only {len(remainder)} remaining trials "
              f"(wanted {args.random_n} random) — using them all", flush=True)
        random_set = remainder
    else:
        random_set = random.sample(remainder, args.random_n)

    all_selected = top_set + random_set
    random.shuffle(all_selected)  # mix for progress display
    print(f"Top {len(top_set)} + random {len(random_set)} = {len(all_selected)} trials", flush=True)

    # Load OOS data
    print("\nLoading OOS data (2022-2023 only)...", flush=True)
    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    all_dates_dict, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates_dict if DATE_LO <= d <= DATE_HI])
    print(f"  {len(dates)} trading days: {dates[0]} to {dates[-1]}", flush=True)

    # Configure simulator
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False

    # Pre-compute V3 candidates from baseline G holds
    print("\nPre-computing V3 candidates from baseline G...", flush=True)
    with open(BASELINE_PATH) as f:
        baseline_full = json.load(f)
    bl_params = dict(baseline_full["params"])
    _oc_set_params(bl_params)
    _disable_adaptive_controls()
    g_holds = {}
    cash = float(STARTING_CASH)
    pre_t0 = time.time()
    for idx, d in enumerate(dates):
        dp = picks_by_date.get(d, [])
        if not dp:
            continue
        is_cash = cash < 25000
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                dp, cash, cash_account=is_cash
            )
        except Exception:
            continue
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0 and st.get("strategy") == "G":
                key = (st.get("ticker"), d)
                et = st.get("entry_time")
                xt = st.get("exit_time")
                if et and xt:
                    g_holds.setdefault(key, []).append((et, xt))
        cash = end_c + (unset if is_cash else 0)
        if (idx + 1) % 100 == 0:
            print(f"  pre-compute [{idx+1}/{len(dates)}] {time.time()-pre_t0:.0f}s  {sum(len(v) for v in g_holds.values())} holds", flush=True)
    print(f"  G hold windows: {sum(len(v) for v in g_holds.values())} in {time.time()-pre_t0:.1f}s", flush=True)

    v3_candidates = compute_v3_candidates(picks_by_date, dates, g_holds)
    print(f"  V3 candidates: {len(v3_candidates)}", flush=True)

    # Run each selected trial OOS
    print(f"\nRunning OOS tests...", flush=True)
    results = []
    t0 = time.time()
    for i, t in enumerate(all_selected):
        trial_start = time.time()
        print(f"  starting trial #{t.number} ({i+1}/{len(all_selected)})...", flush=True)
        try:
            oos = run_oos_trial(t.params, v3_candidates, picks_by_date, dates)
        except Exception as e:
            print(f"  TRIAL #{t.number} FAILED: {e}", flush=True)
            oos = {"g_pnl": 0, "g_pf": 0, "g_wr": 0, "g_n": 0,
                   "v3_pnl": 0, "v3_pf": 0, "v3_n": 0, "combined_pnl": 0}
        print(f"  trial #{t.number} done: {time.time()-trial_start:.0f}s | "
              f"OOS=${oos['combined_pnl']:+,.0f}", flush=True)
        results.append({
            "trial_number": t.number,
            "in_score": t.value,
            "in_g_pnl": t.user_attrs.get("g_pnl", 0),
            "in_g_pf": t.user_attrs.get("g_pf", 0),
            "in_v3_pnl": t.user_attrs.get("v3_pnl", 0),
            "in_combined_pnl": t.user_attrs.get("combined_pnl", 0),
            **oos,
        })
        if (i+1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  --- [{i+1}/{len(all_selected)}] {elapsed:.0f}s elapsed "
                  f"({elapsed/(i+1):.1f}s/trial avg) ---", flush=True)

    # ── Analysis ──
    in_scores = np.array([r["in_score"] for r in results])
    in_g_pnls = np.array([r["in_g_pnl"] for r in results])
    in_combined = np.array([r["in_combined_pnl"] for r in results])
    oos_combined = np.array([r["combined_pnl"] for r in results])
    oos_g_pnls = np.array([r["g_pnl"] for r in results])
    oos_v3_pnls = np.array([r["v3_pnl"] for r in results])

    # Correlation
    valid = ~np.isnan(in_scores) & ~np.isnan(oos_combined) & (np.abs(oos_combined) < 9e12)
    if valid.sum() > 2:
        corr = np.corrcoef(in_scores[valid], oos_combined[valid])[0, 1]
    else:
        corr = 0.0

    valid_pnl = ~np.isnan(in_combined) & ~np.isnan(oos_combined) & (np.abs(oos_combined) < 9e12)
    if valid_pnl.sum() > 2:
        corr_pnl = np.corrcoef(in_combined[valid_pnl], oos_combined[valid_pnl])[0, 1]
    else:
        corr_pnl = 0.0

    # Hit rate: % of top-50 that remain positive OOS
    top_results = [r for r in results if r["trial_number"] in {t.number for t in top_set}]
    non_top = [r for r in results if r["trial_number"] not in {t.number for t in top_set}]

    top_positive_oos = sum(1 for r in top_results if r["combined_pnl"] > 0)
    rand_positive_oos = sum(1 for r in non_top if r["combined_pnl"] > 0)

    # Compare best #564
    best_trial = study.best_trial
    best_result = next((r for r in results if r["trial_number"] == best_trial.number), None)

    print(f"\n{'='*70}")
    print(f"  WIDE FORWARD TEST RESULTS")
    print(f"  Top {len(top_set)} + Random {len(random_set)} trials")
    print(f"  OOS: {DATE_LO} to {DATE_HI} ({len(dates)} trading days)")
    print(f"{'='*70}")

    print(f"\n  ── Correlation ──")
    print(f"  IS score vs OOS combined PnL:  r = {corr:.4f}")
    print(f"  IS combined PnL vs OOS combo:  r = {corr_pnl:.4f}")

    print(f"\n  ── Hit rates ──")
    print(f"  Top-{len(top_set)} OOS positive:  {top_positive_oos}/{len(top_set)} "
          f"({top_positive_oos/len(top_set)*100:.1f}%)")
    print(f"  Random-{len(non_top)} OOS positive: {rand_positive_oos}/{len(non_top)} "
          f"({rand_positive_oos/len(non_top)*100:.1f}%)")

    print(f"\n  ── Best trial (#{best_trial.number}) OOS ──")
    if best_result:
        print(f"  IS score:    ${best_trial.value:,.0f}")
        print(f"  IS G PnL:    ${best_result['in_g_pnl']:+,.0f}")
        print(f"  IS V3 PnL:   ${best_result['in_v3_pnl']:+,.0f}")
        print(f"  OOS G PnL:   ${best_result['g_pnl']:+,.0f}")
        print(f"  OOS V3 PnL:  ${best_result['v3_pnl']:+,.0f}")
        print(f"  OOS combo:   ${best_result['combined_pnl']:+,.0f}")

    print(f"\n  ── Top 10 by OOS PnL ──")
    sorted_oos = sorted(results, key=lambda r: r["combined_pnl"], reverse=True)[:10]
    print(f"  {'#':>5} {'Score(IS)':>14} {'G(IS)':>14} {'G(OOS)':>14} {'V3(OOS)':>10} {'Combo(OOS)':>14}")
    for r in sorted_oos:
        print(f"  {r['trial_number']:>5} ${r['in_score']:>+11,.0f} "
              f"${r['in_g_pnl']:>+11,.0f} ${r['g_pnl']:>+11,.0f} "
              f"${r['v3_pnl']:>+8,.0f} ${r['combined_pnl']:>+11,.0f}")

    print(f"\n  ── Worst 5 by OOS PnL ──")
    worst_oos = sorted(results, key=lambda r: r["combined_pnl"])[:5]
    print(f"  {'#':>5} {'Score(IS)':>14} {'G(IS)':>14} {'G(OOS)':>14} {'V3(OOS)':>10} {'Combo(OOS)':>14}")
    for r in worst_oos:
        print(f"  {r['trial_number']:>5} ${r['in_score']:>+11,.0f} "
              f"${r['in_g_pnl']:>+11,.0f} ${r['g_pnl']:>+11,.0f} "
              f"${r['v3_pnl']:>+8,.0f} ${r['combined_pnl']:>+11,.0f}")

    # Summary table
    print(f"\n  ── Aggregate ──")
    top_avg_g = np.mean([r["g_pnl"] for r in top_results])
    rand_avg_g = np.mean([r["g_pnl"] for r in non_top])
    top_avg_v3 = np.mean([r["v3_pnl"] for r in top_results])
    rand_avg_v3 = np.mean([r["v3_pnl"] for r in non_top])
    top_avg_combo = np.mean([r["combined_pnl"] for r in top_results])
    rand_avg_combo = np.mean([r["combined_pnl"] for r in non_top])
    print(f"{'':>15} {'Top-50 avg':>14} {'Random avg':>14}")
    print(f"{'G PnL':>15} ${top_avg_g:>+11,.0f} ${rand_avg_g:>+11,.0f}")
    print(f"{'V3 PnL':>15} ${top_avg_v3:>+11,.0f} ${rand_avg_v3:>+11,.0f}")
    print(f"{'Combo PnL':>15} ${top_avg_combo:>+11,.0f} ${rand_avg_combo:>+11,.0f}")

    # Save
    os.makedirs("results", exist_ok=True)
    out_data = {
        "n_top": len(top_set),
        "n_random": len(random_set),
        "oos_period": f"{DATE_LO}_{DATE_HI}",
        "correlation_score_vs_oos": round(float(corr), 4),
        "correlation_pnl_vs_oos": round(float(corr_pnl), 4),
        "top_positive_oos_pct": round(top_positive_oos / len(top_set) * 100, 1),
        "rand_positive_oos_pct": round(rand_positive_oos / len(non_top) * 100, 1),
        "top_avg_g_pnl": round(float(top_avg_g), 2),
        "rand_avg_g_pnl": round(float(rand_avg_g), 2),
        "top_avg_v3_pnl": round(float(top_avg_v3), 2),
        "rand_avg_v3_pnl": round(float(rand_avg_v3), 2),
        "top_avg_combo_pnl": round(float(top_avg_combo), 2),
        "rand_avg_combo_pnl": round(float(rand_avg_combo), 2),
        "top_oos": {r["trial_number"]: {
            "in_score": r["in_score"],
            "in_g_pnl": r["in_g_pnl"],
            "in_combined_pnl": r["in_combined_pnl"],
            "oos_g_pnl": r["g_pnl"],
            "oos_v3_pnl": r["v3_pnl"],
            "oos_combined_pnl": r["combined_pnl"],
        } for r in top_results},
        "random_oos": {r["trial_number"]: {
            "in_score": r["in_score"],
            "in_g_pnl": r["in_g_pnl"],
            "in_combined_pnl": r["in_combined_pnl"],
            "oos_g_pnl": r["g_pnl"],
            "oos_v3_pnl": r["v3_pnl"],
            "oos_combined_pnl": r["combined_pnl"],
        } for r in non_top},
    }
    with open(args.out, "w") as f:
        json.dump(out_data, f, indent=2, default=str)
    print(f"\nWrote {args.out}")

    # Interpretation
    print(f"\n  ── Verdict ──")
    if corr > 0.3:
        print(f"  ✅ Strong IS→OOS correlation ({corr:.3f}) — signal generalizes")
    elif corr > 0.15:
        print(f"  🟡 Moderate IS→OOS correlation ({corr:.3f}) — partial signal")
    else:
        print(f"  🟠 Weak IS→OOS correlation ({corr:.3f}) — likely overfit")

    if best_result and best_result["combined_pnl"] > 0:
        print(f"  ✅ Best trial #{best_trial.number} is positive OOS (${best_result['combined_pnl']:+,.0f})")
    else:
        print(f"  ❌ Best trial #{best_trial.number} loses money OOS")

    pct = top_positive_oos / len(top_set) * 100
    if pct >= 60:
        print(f"  ✅ Top trials hit rate OOS: {pct:.0f}%")
    elif pct >= 40:
        print(f"  🟡 Top trials hit rate OOS: {pct:.0f}%")
    else:
        print(f"  ❌ Top trials hit rate OOS: {pct:.0f}%")


if __name__ == "__main__":
    main()
