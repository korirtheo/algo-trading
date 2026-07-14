"""
Wide forward test of G+V3 joint #564 best params on 2022-2023 OOS data.

Runs the same G-only + V3 overlay simulation from optuna_g_v3_joint.py
but on 2022-01-01 to 2023-12-31 — never seen during training.
"""
import argparse
import json
import math
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import test_green_candle_combined as tgc
from test_full import load_all_picks
from optimize_combined import set_strategy_params as _oc_set_params
from optimize_combined import _build_param_snapshot, _param_lock

STARTING_CASH = 25_000
POSITION_PCT = 0.30
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
BASELINE_PATH = "config/trial_g511_l626_v3_overlay.json"

# OOS data dirs — only 2022 + 2023, never seen in training (2024-2026)
DATA_DIRS = ["stored_data_2022", "stored_data_2023"]
DATE_LO = "2022-01-01"
DATE_HI = "2023-12-31"


def _disable_adaptive_controls():
    tgc.MIN_PRICE = 0.0
    tgc.MAX_MODELED_SLIP_BP = 0.0
    tgc.MAX_CUM_DVOL_AT_ENTRY_M = 0.0
    tgc.MIN_ATR_PCT = 0.0
    tgc.MIN_FAVORABILITY_THRESHOLD = 0.0
    for s in ALL_STRATS:
        setattr(tgc, f"{s.upper()}_PARTICIPATION_CAP", 0.0)
    tgc.NEWS_FILTER_ENABLED = False


def load_g_holds(picks_by_date, dates, g_params):
    """Run G-only baseline to compute hold windows (same as Phase 1)."""
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False

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

    g_holds = {}
    cash = float(STARTING_CASH)
    for d in dates:
        dp = picks_by_date.get(d, [])
        if not dp:
            continue
        is_cash = cash < 25000  # MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                dp, cash, cash_account=is_cash, params=snapshot
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
    return g_holds


def compute_v3_candidates(picks_by_date, dates, g_holds):
    """Pre-compute V3 trade candidates from G hold windows."""
    counters = {"no_data": 0, "g_exited_after_eod": 0, "no_hit": 0, "fired": 0}
    candidates = []
    for d in dates:
        for p in picks_by_date.get(d, []):
            mh = p.get("market_hour_candles")
            if mh is None or len(mh) < 2:
                counters["no_data"] += 1
                continue
            day_open = float(mh.iloc[0]["Open"])
            bar0_close = float(mh.iloc[0]["Close"])
            bar0_red = bar0_close <= day_open
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
                    counters["g_exited_after_eod"] += 1
                    continue
                scan_start = new_start

            entry_idx = entry_price = entry_ts = None
            for i in range(scan_start, len(mh)):
                c = float(mh.iloc[i]["Close"])
                if c > day_open:
                    entry_idx, entry_price, entry_ts = i, c, mh.index[i]
                    break
            if entry_idx is None or entry_price is None or entry_price <= 0:
                counters["no_hit"] += 1
                continue
            bars_after = mh.iloc[entry_idx + 1:]
            if len(bars_after) == 0:
                continue
            counters["fired"] += 1
            candidates.append({
                "ticker": p["ticker"], "date": d,
                "entry_ts": entry_ts, "entry_price": entry_price,
                "mh": mh, "bars_after": bars_after,
            })

    print(f"  V3 candidates: {len(candidates)} ({counters['fired']} fired)")
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
            exit_price = float(row["Close"])
            exit_ts = ts
            break
        c_high = float(row["High"])
        c_low = float(row["Low"])
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


def run_forward_test(best_config):
    """Run the full forward test: G sim + V3 overlay on OOS period."""
    g_params = dict(best_config["params"])
    v3_params = dict(best_config["v3_overlay"])

    print("=" * 70)
    print(f"  FORWARD TEST: {best_config['label']}")
    print(f"  Period: {DATE_LO} to {DATE_HI}")
    print("=" * 70)
    print()

    # Load data
    print("Loading OOS data (2022-2023 only)...")
    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    all_dates_dict, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates_dict if DATE_LO <= d <= DATE_HI])
    print(f"  {len(dates)} trading days: {dates[0]} to {dates[-1]}")

    # Pre-configure simulator
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False

    # Phase A: G baseline (to get hold windows)
    print("\nPhase A: Running G baseline for hold windows...")
    g_holds = load_g_holds(picks_by_date, dates, g_params)
    n_holds = sum(len(v) for v in g_holds.values())
    print(f"  G hold windows: {n_holds} across {len(g_holds)} pairs")

    # Phase B: Pre-compute V3 candidates
    print("\nPhase B: Pre-computing V3 candidates...")
    v3_candidates = compute_v3_candidates(picks_by_date, dates, g_holds)

    # Phase C: Run full G backtest with same params
    print("\nPhase C: Running full G backtest...")
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
    g_trades = []
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
                g_trades.append({
                    "date": d, "ticker": st["ticker"],
                    "entry": st.get("entry_price"), "exit": st.get("exit_price"),
                    "pnl": pnl, "reason": st.get("exit_reason"),
                })
        cash = end_c + (unset if is_cash else 0)

    # Phase D: V3 overlay
    print("Phase D: Running V3 overlay...")
    v3_pnls = []
    for c in v3_candidates:
        pnl = _v3_sim(v3_params, c)
        if pnl != 0.0:
            v3_pnls.append(pnl)

    # ── Results ──
    total_cash = cash + sum(v3_pnls)
    total_pnl = total_cash - STARTING_CASH
    total_mult = total_cash / STARTING_CASH

    def calc_stats(pnls):
        n = len(pnls)
        if n == 0:
            return 0, 0, 0, 0, 0, 0
        total = sum(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gw = sum(wins) if wins else 0
        gl = abs(sum(losses)) if losses else 1e-9
        pf = gw / gl
        wr = len(wins) / n * 100
        avg = total / n
        best = max(pnls) if wins else 0
        worst = min(pnls) if losses else 0
        return n, total, pf, wr, avg, best, worst, gw, gl

    print(f"\n{'='*70}")
    print(f"  RESULTS: OOS 2022-2023")
    print(f"  Starting Cash: ${STARTING_CASH:,.2f}")
    print(f"  Ending Cash:   ${total_cash:,.2f}")
    print(f"  Total PnL:     ${total_pnl:+,.2f}  ({total_mult:.2f}x)")
    print(f"{'='*70}")

    g_n, g_tot, g_pf, g_wr, g_avg, g_best, g_worst, g_gw, g_gl = calc_stats(g_pnls)
    v3_n, v3_tot, v3_pf, v3_wr, v3_avg, v3_best, v3_worst, v3_gw, v3_gl = calc_stats(v3_pnls)

    print()
    print(f"{'─'*70}")
    print(f"{'':>12} {'Trades':>8} {'Win%':>7} {'PF':>8} {'Total PnL':>12} {'Avg':>10} {'Best':>10} {'Worst':>10}")
    print(f"{'─'*70}")
    print(f"{'G':>12} {g_n:>8} {g_wr:>6.1f}% {g_pf:>7.2f} ${g_tot:>+9,.0f} ${g_avg:>+8,.2f} ${g_best:>+8,.2f} ${g_worst:>+8,.2f}")
    print(f"{'V3':>12} {v3_n:>8} {v3_wr:>6.1f}% {v3_pf:>7.2f} ${v3_tot:>+9,.0f} ${v3_avg:>+8,.2f} ${v3_best:>+8,.2f} ${v3_worst:>+8,.2f}")
    print(f"{'Combined':>12} {g_n+v3_n:>8} {'':>7} {'':>8} ${total_pnl:>+9,.0f} {'':>10} {'':>10} {'':>10}")
    print(f"{'─'*70}")

    # Compare with #511 baseline
    print(f"\n  vs deployed #511: G={g_tot:+,.0f} V3={v3_tot:+,.0f} combo={total_pnl:+,.0f}")
    print(f"  G WR={g_wr:.1f}% PF={g_pf:.2f} n={g_n} | V3 PF={v3_pf:.2f} n={v3_n}")
    print()

    # Save
    out = "results/forward_test_g_v3_2022_2023.json"
    os.makedirs("results", exist_ok=True)
    result = {
        "label": best_config["label"],
        "period": f"{DATE_LO}_{DATE_HI}",
        "starting_cash": STARTING_CASH,
        "end_cash": round(total_cash, 2),
        "total_pnl": round(total_pnl, 2),
        "multiplier": round(total_mult, 4),
        "g": {"n": g_n, "pnl": round(g_tot, 2), "pf": round(g_pf, 3), "wr": round(g_wr, 1)},
        "v3": {"n": v3_n, "pnl": round(v3_tot, 2), "pf": round(v3_pf, 3), "wr": round(v3_wr, 1)},
        "g_trades": g_trades,
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G+V3 forward test on 2022-2023")
    parser.add_argument("--config", default="config/trial_g_v3_joint_best.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print(f"Config: {config['label']}")
    print(f"  G params:  min_gap={config['params']['g_min_gap_pct']}%, "
          f"target={config['params']['g_target_pct']}%, "
          f"stop={config['params']['g_stop_pct']}%, "
          f"time={config['params']['g_time_limit_min']}min, "
          f"2nd_green={config['params']['g_require_2nd_green']}, "
          f"2nd_NH={config['params']['g_require_2nd_new_high']}, "
          f"trail={config['params']['g_trail_pct']}%, "
          f"trail_act={config['params']['g_trail_activate_pct']}%")
    print(f"  V3 params: target={config['v3_overlay']['target_pct']}%, "
          f"stop={config['v3_overlay']['stop_pct']}%, "
          f"time={config['v3_overlay']['time_limit_min']}min, "
          f"trail={config['v3_overlay']['trail_pct']}%, "
          f"trail_act={config['v3_overlay']['trail_activate_pct']}%")

    run_forward_test(config)
