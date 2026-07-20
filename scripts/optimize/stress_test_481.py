"""
Stress Test: Trial #481 on 2026 OOS (Mar-Jun, 60 blind days)
Flat $10,000/day capital. Full per-trade instrumentation.

Note: flat $10K/day means cash resets each day. With FULL_BALANCE_SIZING=True
the simulator deploys the whole balance per trade, so we get ~1 trade/day when
only one setup triggers. This is the correct stress-test behaviour: we want to
see raw per-trade quality without compounding distortion.
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params, _build_param_snapshot
from test_full import load_all_picks

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DAILY_CAPITAL    = 10_000.0
MARGIN_THRESHOLD = 25_000.0

DATA_DIRS = [
    "stored_data_mar_may_2026",
    "stored_data_jun_2026",
    "stored_data_oos",
]
OOS_CUTOFF = "2026-03-01"

TRIAL_538_PARAMS = {
    "g_min_gap_pct":          10.0,
    "g_partial_sell_pct":     25.0,
    "g_require_2nd_green":    True,
    "g_require_2nd_new_high": True,
    "g_stop_pct":             26.0,
    "g_target2_pct":          60.0,
    "g_target_pct":           15.0,
    "g_time_limit_min":       12,
    "g_use_trail":            False,
    "l_earliest_candle":      24,
    "l_latest_candle":        165,
    "l_max_float":            25000000,
    "l_min_gap":              25,
    "l_min_price_accel_pct":  0.5,
    "l_partial_sell_pct":     75.0,
    "l_stop_pct":             18.0,
    "l_tier1_target1_pct":    40.0,
    "l_tier1_target2_pct":    30.0,
    "l_tier2_target1_pct":    30.0,
    "l_tier2_target2_pct":    55.0,
    "l_tier3_target1_pct":    25.0,
    "l_tier3_target2_pct":    40.0,
    "l_trail_activate_pct":   1.0,
    "l_trail_pct":            1.0,
    "l_use_trail":            True,
}

# ─────────────────────────────────────────────────────────────────────────────
# Patch _multi_window_effective_volume to record vol components per (ticker, ts)
# so we can look them up after each trade closes.
# ─────────────────────────────────────────────────────────────────────────────
_vol_log = {}  # (ticker, str(ts)) -> dict

_orig_mw = tgc._multi_window_effective_volume

def _patched_mw(mh, ts, fill_price):
    v_eff, v_2min, v_local, v_regime = _orig_mw(mh, ts, fill_price)
    # Identify ticker from mh object name (column not available here, use id)
    _vol_log[id(mh)] = {
        "ts":       ts,
        "v_eff":    v_eff,
        "v_2min":   v_2min,
        "v_local":  v_local,
        "v_regime": v_regime,
    }
    return v_eff, v_2min, v_local, v_regime

tgc._multi_window_effective_volume = _patched_mw


def run_stress_test():
    # ── 1. Load OOS data ──────────────────────────────────────────────────────
    dirs = [d for d in DATA_DIRS if Path(d).exists()]
    all_dates, daily_picks = load_all_picks(dirs)
    oos_dates = sorted([d for d in all_dates if d >= OOS_CUTOFF])
    oos_picks  = {d: daily_picks[d] for d in oos_dates if d in daily_picks}
    print(f"OOS window: {oos_dates[0]} to {oos_dates[-1]}  ({len(oos_dates)} days)")

    # ── 2. Set params ─────────────────────────────────────────────────────────
    with open("config/trial_w21b_511_deploy.json") as f:
        base_cfg = json.load(f)
    merged = base_cfg["params"].copy()
    merged.update(TRIAL_538_PARAMS)
    set_strategy_params(merged)
    tgc.USE_DYNAMIC_SLIPPAGE     = True
    tgc.USE_MULTIWINDOW_SLIPPAGE  = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K            = 3.0
    tgc.VOL_CAP_PCT              = 5.0
    tgc.MAX_2MIN_PARTICIPATION   = 0.15
    tgc.MAX_REGIME_PARTICIPATION  = 0.08
    snapshot = _build_param_snapshot()

    # ── 3. Run day by day ─────────────────────────────────────────────────────
    all_trades    = []
    daily_results = []

    for date in oos_dates:
        picks = oos_picks.get(date, [])
        if not picks:
            continue

        _vol_log.clear()
        cash = DAILY_CAPITAL
        cash_account = cash < MARGIN_THRESHOLD

        states, ending_cash, unsettled, _ = tgc.simulate_day_combined(
            picks, cash, cash_account, params=snapshot
        )

        day_pnl = day_wins = day_losses = day_trades = 0

        for st in states:
            if st.get("exit_reason") is None:
                continue

            pnl      = st["pnl"]
            cost     = st["position_cost"]
            strategy = st.get("strategy", "?")
            ticker   = st.get("ticker", "?")
            entry_ts = st.get("entry_time")

            # Volume reconstruction: use mh reference still on state
            mh = st.get("mh")
            v_2min = v_6min = v_10min = v_eff = 0.0
            cum_dvol = cum_vol_shares = 0.0
            part_veff = part_cum = 0.0

            if mh is not None and entry_ts is not None and cost > 0:
                try:
                    # Use the saved fill price approximation: entry_price is
                    # cleared on close, but we can back-calculate from cost/shares
                    # Alternatively look up from vol_log by mh id
                    vl = _vol_log.get(id(mh))
                    ep_approx = st.get("entry_price")  # None if closed

                    # Best effort: recompute from mh at entry_ts
                    # entry_price may be None after close; use signal_price as proxy
                    sig_price = st.get("signal_price") or 0
                    if ep_approx is None and sig_price > 0:
                        ep_approx = sig_price  # slight over-estimate (pre-slip)

                    if ep_approx and ep_approx > 0:
                        v_eff, v_2min, v_6min_raw, v_10min_raw = _orig_mw(mh, entry_ts, ep_approx)
                        v_6min   = v_6min_raw
                        v_10min  = v_10min_raw
                        pre_bars = mh.loc[mh.index <= entry_ts]
                        cum_vol_shares = float(pre_bars["Volume"].sum())
                        cum_dvol       = cum_vol_shares * ep_approx
                        if v_eff > 0:
                            part_veff = cost / v_eff * 100
                        if cum_dvol > 0:
                            part_cum  = cost / cum_dvol * 100
                except Exception:
                    pass

            all_trades.append({
                "date":              date,
                "ticker":            ticker,
                "strategy":          strategy,
                "pnl":               pnl,
                "position_cost":     cost,
                "exit_reason":       st.get("exit_reason", "?"),
                "v_2min":            v_2min,
                "v_6min":            v_6min,
                "v_10min":           v_10min,
                "v_eff_adj":         v_eff,
                "cum_dvol_at_entry": cum_dvol,
                "cum_vol_shares":    cum_vol_shares,
                "part_veff_pct":     part_veff,
                "part_cum_pct":      part_cum,
                "vol_capped":        st.get("vol_capped", False),
            })

            day_pnl    += pnl
            day_trades += 1
            if pnl > 0:
                day_wins += 1
            else:
                day_losses += 1

        daily_results.append({
            "date":    date,
            "pnl":     day_pnl,
            "trades":  day_trades,
            "wins":    day_wins,
            "losses":  day_losses,
            "green":   day_pnl > 0,
        })

    return all_trades, daily_results


def print_report(trades, daily):
    df  = pd.DataFrame(trades)
    ddf = pd.DataFrame(daily)

    print()
    print("=" * 72)
    print("STRESS TEST: Trial #538  |  Flat $10,000/day  |  2026 OOS (Mar-Jun)")
    print("=" * 72)

    total_days   = len(ddf)
    green_days   = int(ddf["green"].sum())
    red_days     = total_days - green_days
    total_pnl    = ddf["pnl"].sum()
    avg_daily    = ddf["pnl"].mean()
    total_trades = len(df)

    wins   = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    wr     = len(wins) / total_trades * 100 if total_trades else 0
    avg_win  = wins["pnl"].mean()   if len(wins)   else 0.0
    avg_loss = losses["pnl"].mean() if len(losses) else 0.0
    gross_w  = wins["pnl"].sum()
    gross_l  = abs(losses["pnl"].sum())
    pf       = gross_w / gross_l if gross_l > 0 else 99.0

    # Drawdown on cumulative PnL curve
    cum   = ddf["pnl"].cumsum()
    peak  = cum.cummax()
    dd    = peak - cum
    max_dd = dd.max()
    max_dd_date = ddf.loc[dd.idxmax(), "date"] if not dd.empty else "N/A"

    # Win/loss streaks
    streak_win = streak_loss = cur_win = cur_loss = 0
    for g in ddf["green"]:
        if g:
            cur_win += 1
            streak_win  = max(streak_win,  cur_win)
            cur_loss = 0
        else:
            cur_loss += 1
            streak_loss = max(streak_loss, cur_loss)
            cur_win  = 0
    streak_loss = max(streak_loss, cur_loss)

    # Return on daily capital
    total_return_pct = total_pnl / (DAILY_CAPITAL * total_days) * 100

    print(f"\n  OVERVIEW")
    print(f"  {'Trading days':<32} {total_days}")
    print(f"  {'Total PnL':<32} ${total_pnl:,.2f}")
    print(f"  {'Avg PnL/day':<32} ${avg_daily:,.2f}")
    print(f"  {'Return on deployed capital':<32} {total_return_pct:.1f}%  (vs ${DAILY_CAPITAL:,.0f}/day x {total_days} days)")
    print(f"  {'Green days':<32} {green_days}  ({100*green_days/total_days:.1f}%)")
    print(f"  {'Red days':<32} {red_days}  ({100*red_days/total_days:.1f}%)")
    print(f"  {'Max drawdown (cum PnL)':<32} ${max_dd:,.2f}  (at {max_dd_date})")
    print(f"  {'Longest win streak':<32} {streak_win} days")
    print(f"  {'Longest loss streak':<32} {streak_loss} days")

    print(f"\n  TRADES")
    print(f"  {'Total trades':<32} {total_trades}")
    print(f"  {'Avg trades/day':<32} {total_trades/total_days:.2f}")
    print(f"  {'Win rate':<32} {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  {'Profit factor':<32} {pf:.2f}")
    print(f"  {'Avg win':<32} ${avg_win:,.2f}")
    print(f"  {'Avg loss':<32} ${avg_loss:,.2f}")
    print(f"  {'Avg win / avg loss ratio':<32} {abs(avg_win/avg_loss):.2f}x")
    print(f"  {'Gross win':<32} ${gross_w:,.2f}")
    print(f"  {'Gross loss':<32} ${gross_l:,.2f}")
    print(f"  {'Avg position size':<32} ${df['position_cost'].mean():,.2f}")
    print(f"  {'Max position size':<32} ${df['position_cost'].max():,.2f}")
    print(f"  {'Min position size':<32} ${df['position_cost'].min():,.2f}")

    # Exit reason breakdown
    print(f"\n  EXIT REASONS")
    for reason, grp in df.groupby("exit_reason"):
        r_wins = (grp["pnl"] > 0).sum()
        print(f"  {'  '+reason:<32} {len(grp):3d} trades  "
              f"WR={100*r_wins/len(grp):.0f}%  "
              f"PnL=${grp['pnl'].sum():,.2f}  avg=${grp['pnl'].mean():,.2f}")

    # By strategy
    print(f"\n  BY STRATEGY")
    print(f"  {'Strat':>5}  {'n':>4}  {'WR':>6}  {'PF':>5}  {'Total PnL':>11}  {'AvgWin':>8}  {'AvgLoss':>8}  {'AvgPos':>8}")
    print(f"  {'-'*67}")
    for s, grp in df.groupby("strategy"):
        sw = grp[grp["pnl"] > 0]
        sl = grp[grp["pnl"] <= 0]
        s_pf = sw["pnl"].sum() / abs(sl["pnl"].sum()) if len(sl) and sl["pnl"].sum() != 0 else 99.0
        s_wr = 100 * len(sw) / len(grp)
        s_aw = sw["pnl"].mean() if len(sw) else 0.0
        s_al = sl["pnl"].mean() if len(sl) else 0.0
        s_ap = grp["position_cost"].mean()
        print(f"  {s:>5}  {len(grp):>4}  {s_wr:>5.1f}%  {s_pf:>5.2f}  "
              f"${grp['pnl'].sum():>9,.2f}  ${s_aw:>7,.2f}  ${s_al:>7,.2f}  ${s_ap:>7,.2f}")

    # Liquidity section
    has_vol = df[df["v_2min"] > 0].copy()
    print(f"\n  LIQUIDITY & PARTICIPATION  (multiwindow slippage model, {len(has_vol)}/{total_trades} trades with vol data)")
    if len(has_vol) > 0:
        def _row(label, col, fmt="$"):
            s = has_vol[col].dropna()
            if len(s) == 0:
                return
            if fmt == "$":
                print(f"  {label:<38} mean=${s.mean():>10,.0f}  median=${s.median():>10,.0f}  p95=${s.quantile(0.95):>10,.0f}")
            else:
                print(f"  {label:<38} mean={s.mean():>9.2f}%  median={s.median():>9.2f}%  p95={s.quantile(0.95):>9.2f}%")

        _row("Position deployed ($)",         "position_cost")
        _row("2-min liquidity window ($)",     "v_2min")
        _row("6-min liquidity window ($)",     "v_6min")
        _row("10-min liquidity window ($)",    "v_10min")
        _row("V_eff_adj (binding cap, $)",     "v_eff_adj")
        _row("Cum dollar vol at entry ($)",    "cum_dvol_at_entry")
        _row("Participation vs V_eff (%)",     "part_veff_pct",  "%")
        _row("Participation vs cum dvol (%)",  "part_cum_pct",   "%")

        capped = int(has_vol["vol_capped"].sum())
        print(f"  {'Vol-capped trades':<38} {capped}  ({100*capped/len(has_vol):.1f}%)")

        # Top 10 largest deployments
        top10 = has_vol.nlargest(10, "position_cost")
        print(f"\n  TOP 10 LARGEST POSITIONS")
        hdr = f"  {'Date':>10}  {'Ticker':>8}  {'S':>2}  {'Deployed':>9}  {'V_2min':>9}  {'V_6min':>9}  {'V_10min':>9}  {'V_eff':>9}  {'CumDVol':>11}  {'Part%':>6}  {'PnL':>8}  Cap"
        print(hdr)
        print(f"  {'-'*110}")
        for _, r in top10.iterrows():
            cap = "Y" if r["vol_capped"] else "N"
            print(f"  {r['date']:>10}  {r['ticker']:>8}  {r['strategy']:>2}  "
                  f"${r['position_cost']:>8,.0f}  "
                  f"${r['v_2min']:>8,.0f}  ${r['v_6min']:>8,.0f}  ${r['v_10min']:>8,.0f}  "
                  f"${r['v_eff_adj']:>8,.0f}  ${r['cum_dvol_at_entry']:>10,.0f}  "
                  f"{r['part_veff_pct']:>5.1f}%  ${r['pnl']:>7,.0f}  {cap}")

    # Daily breakdown
    print(f"\n  DAILY PnL BREAKDOWN")
    print(f"  {'Date':>10}  {'PnL':>10}  {'Trd':>3}  {'W/L':>5}  {'Cumul PnL':>12}")
    print(f"  {'-'*50}")
    cumul = 0.0
    for _, r in ddf.iterrows():
        cumul += r["pnl"]
        sign  = "+" if r["pnl"] >= 0 else "-"
        wl    = f"{int(r['wins'])}W/{int(r['losses'])}L"
        print(f"  {r['date']:>10}  {sign}${abs(r['pnl']):>9,.2f}  {int(r['trades']):>3}  {wl:>5}  ${cumul:>11,.2f}")

    print(f"\n  {'='*72}")
    print(f"  TOTAL ${total_pnl:,.2f} | {total_days} days | {total_trades} trades | {green_days}G {red_days}R | max DD ${max_dd:,.2f}")
    print(f"  {'='*72}")


if __name__ == "__main__":
    trades, daily = run_stress_test()
    print_report(trades, daily)
