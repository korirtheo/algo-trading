"""
Full detailed backtest: Trial #538 on 2026 OOS (Mar-Jun)
$25K rolling capital (standard), multiwindow slippage.
Reports: total trades, avg daily gain, avg win/loss, green/red days,
max drawdown, avg pnl/day, top deployments with 2/6/10min participation,
and anything else interesting.
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

STARTING_CASH    = 25_000.0
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

_orig_mw = tgc._multi_window_effective_volume

def _patched_mw(mh, ts, fill_price):
    v_eff, v_2min, v_local, v_regime = _orig_mw(mh, ts, fill_price)
    _vol_log[id(mh)] = {"v_eff": v_eff, "v_2min": v_2min,
                        "v_local": v_local, "v_regime": v_regime, "ts": ts}
    return v_eff, v_2min, v_local, v_regime

_vol_log = {}
tgc._multi_window_effective_volume = _patched_mw


def run_backtest():
    dirs = [d for d in DATA_DIRS if Path(d).exists()]
    all_dates, daily_picks = load_all_picks(dirs)
    oos_dates = sorted([d for d in all_dates if d >= OOS_CUTOFF])
    oos_picks  = {d: daily_picks[d] for d in oos_dates if d in daily_picks}
    print(f"OOS window: {oos_dates[0]} to {oos_dates[-1]}  ({len(oos_dates)} days)")

    with open("config/trial_w21b_511_deploy.json") as f:
        base_cfg = json.load(f)
    merged = base_cfg["params"].copy()
    merged.update(TRIAL_538_PARAMS)
    set_strategy_params(merged)
    tgc.USE_DYNAMIC_SLIPPAGE      = True
    tgc.USE_MULTIWINDOW_SLIPPAGE   = True
    tgc.USE_VOLATILITY_ADJUSTMENT  = True
    tgc.SLIP_IMPACT_K             = 3.0
    tgc.VOL_CAP_PCT               = 5.0
    tgc.MAX_2MIN_PARTICIPATION    = 0.15
    tgc.MAX_REGIME_PARTICIPATION   = 0.08
    snapshot = _build_param_snapshot()

    cash      = STARTING_CASH
    unsettled = 0.0
    all_trades    = []
    daily_results = []

    for date in oos_dates:
        picks = oos_picks.get(date, [])
        cash += unsettled
        unsettled = 0.0

        if not picks:
            daily_results.append({"date": date, "pnl": 0.0, "trades": 0,
                                   "wins": 0, "losses": 0, "green": True,
                                   "equity_start": cash, "equity_end": cash})
            continue

        cash_account = cash < MARGIN_THRESHOLD
        equity_start = cash
        _vol_log.clear()

        states, new_cash, new_unsettled, _ = tgc.simulate_day_combined(
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
            mh       = st.get("mh")

            v_2min = v_6min = v_10min = v_eff = 0.0
            cum_dvol = cum_vol_shares = 0.0
            part_veff = part_cum = 0.0

            if mh is not None and entry_ts is not None and cost > 0:
                try:
                    sig_price = st.get("signal_price") or 0
                    ep = st.get("entry_price") or sig_price
                    if ep and ep > 0:
                        v_eff, v_2min, v_6min, v_10min = _orig_mw(mh, entry_ts, ep)
                        pre = mh.loc[mh.index <= entry_ts]
                        cum_vol_shares = float(pre["Volume"].sum())
                        cum_dvol       = cum_vol_shares * ep
                        if v_eff > 0:
                            part_veff = cost / v_eff * 100
                        if cum_dvol > 0:
                            part_cum  = cost / cum_dvol * 100
                except Exception:
                    pass

            all_trades.append({
                "date": date, "ticker": ticker, "strategy": strategy,
                "pnl": pnl, "position_cost": cost,
                "exit_reason": st.get("exit_reason", "?"),
                "v_2min": v_2min, "v_6min": v_6min, "v_10min": v_10min,
                "v_eff_adj": v_eff,
                "cum_dvol_at_entry": cum_dvol,
                "cum_vol_shares": cum_vol_shares,
                "part_veff_pct": part_veff,
                "part_cum_pct": part_cum,
                "vol_capped": st.get("vol_capped", False),
            })

            day_pnl    += pnl
            day_trades += 1
            if pnl > 0:
                day_wins += 1
            else:
                day_losses += 1

        cash      = new_cash
        unsettled = new_unsettled
        equity_end = cash + unsettled

        daily_results.append({
            "date": date, "pnl": day_pnl, "trades": day_trades,
            "wins": day_wins, "losses": day_losses, "green": day_pnl > 0,
            "equity_start": equity_start, "equity_end": equity_end,
        })

    return all_trades, daily_results


def print_report(trades, daily):
    df  = pd.DataFrame(trades)
    ddf = pd.DataFrame(daily)

    print()
    print("=" * 76)
    print("BACKTEST: Trial #538  |  $25K Rolling Capital  |  2026 OOS (Mar-Jun)")
    print("=" * 76)

    total_days   = len(ddf)
    active_days  = int((ddf["trades"] > 0).sum())
    green_days   = int(ddf["green"].sum())
    red_days     = total_days - green_days
    total_pnl    = ddf["pnl"].sum()
    avg_daily    = ddf["pnl"].mean()
    avg_active   = ddf[ddf["trades"] > 0]["pnl"].mean()
    total_trades = len(df)

    wins   = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]
    wr     = len(wins) / total_trades * 100 if total_trades else 0
    avg_win  = wins["pnl"].mean()   if len(wins)   else 0.0
    avg_loss = losses["pnl"].mean() if len(losses) else 0.0
    gross_w  = wins["pnl"].sum()
    gross_l  = abs(losses["pnl"].sum())
    pf       = gross_w / gross_l if gross_l > 0 else 99.0

    # Equity curve for drawdown
    equity = ddf["equity_end"].values
    peak   = np.maximum.accumulate(equity)
    dd_abs = peak - equity
    dd_pct = dd_abs / peak * 100
    max_dd_abs  = dd_abs.max()
    max_dd_pct  = dd_pct.max()
    max_dd_date = ddf.iloc[dd_abs.argmax()]["date"]

    # Streaks
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

    final_equity = ddf["equity_end"].iloc[-1]
    total_return = (final_equity - STARTING_CASH) / STARTING_CASH * 100

    print(f"\n  OVERVIEW")
    print(f"  {'Starting capital':<34} ${STARTING_CASH:,.2f}")
    print(f"  {'Final equity':<34} ${final_equity:,.2f}")
    print(f"  {'Total PnL':<34} ${total_pnl:,.2f}  ({total_return:.1f}% return)")
    print(f"  {'Avg PnL/day (all 60 days)':<34} ${avg_daily:,.2f}")
    print(f"  {'Avg PnL/day (active days only)':<34} ${avg_active:,.2f}")
    print(f"  {'Total trading days':<34} {total_days}  (active: {active_days})")
    print(f"  {'Green days':<34} {green_days}  ({100*green_days/total_days:.1f}%)")
    print(f"  {'Red days':<34} {red_days}  ({100*red_days/total_days:.1f}%)")
    print(f"  {'Max drawdown ($)':<34} ${max_dd_abs:,.2f}  ({max_dd_pct:.1f}% of equity peak, at {max_dd_date})")
    print(f"  {'Longest win streak':<34} {streak_win} days")
    print(f"  {'Longest loss streak':<34} {streak_loss} days")

    print(f"\n  TRADES")
    print(f"  {'Total trades':<34} {total_trades}")
    print(f"  {'Avg trades/day (all)':<34} {total_trades/total_days:.2f}")
    print(f"  {'Avg trades/day (active)':<34} {total_trades/active_days:.2f}")
    print(f"  {'Win rate':<34} {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
    print(f"  {'Profit factor':<34} {pf:.2f}")
    print(f"  {'Avg win':<34} ${avg_win:,.2f}")
    print(f"  {'Avg loss':<34} ${avg_loss:,.2f}")
    print(f"  {'Avg win / avg loss ratio':<34} {abs(avg_win/avg_loss):.2f}x")
    print(f"  {'Gross win':<34} ${gross_w:,.2f}")
    print(f"  {'Gross loss':<34} ${gross_l:,.2f}")
    print(f"  {'Avg position size':<34} ${df['position_cost'].mean():,.2f}")
    print(f"  {'Max position size':<34} ${df['position_cost'].max():,.2f}")
    print(f"  {'Median position size':<34} ${df['position_cost'].median():,.2f}")

    # Exit reason breakdown
    print(f"\n  EXIT REASONS")
    print(f"  {'Reason':<20} {'n':>4}  {'WR':>6}  {'Total PnL':>12}  {'Avg PnL':>10}")
    print(f"  {'-'*56}")
    for reason, grp in df.groupby("exit_reason"):
        r_wins = (grp["pnl"] > 0).sum()
        print(f"  {reason:<20} {len(grp):>4}  {100*r_wins/len(grp):>5.0f}%  "
              f"${grp['pnl'].sum():>10,.2f}  ${grp['pnl'].mean():>8,.2f}")

    # By strategy
    print(f"\n  BY STRATEGY")
    print(f"  {'Strat':>5}  {'n':>4}  {'WR':>6}  {'PF':>5}  {'Total PnL':>12}  {'AvgWin':>8}  {'AvgLoss':>8}  {'AvgPos':>9}")
    print(f"  {'-'*70}")
    for s, grp in df.groupby("strategy"):
        sw = grp[grp["pnl"] > 0]
        sl = grp[grp["pnl"] <= 0]
        s_pf = sw["pnl"].sum() / abs(sl["pnl"].sum()) if len(sl) and sl["pnl"].sum() != 0 else 99.0
        print(f"  {s:>5}  {len(grp):>4}  {100*len(sw)/len(grp):>5.1f}%  {s_pf:>5.2f}  "
              f"${grp['pnl'].sum():>10,.2f}  ${sw['pnl'].mean() if len(sw) else 0:>7,.2f}  "
              f"${sl['pnl'].mean() if len(sl) else 0:>7,.2f}  ${grp['position_cost'].mean():>8,.2f}")

    # Liquidity
    hv = df[df["v_2min"] > 0].copy()
    print(f"\n  LIQUIDITY & PARTICIPATION  (multiwindow slippage, {len(hv)}/{total_trades} trades with vol data)")
    if len(hv) > 0:
        def _row(label, col, pct=False):
            s = hv[col].dropna()
            if len(s) == 0:
                return
            if pct:
                print(f"  {label:<40} mean={s.mean():>8.2f}%  median={s.median():>8.2f}%  p95={s.quantile(0.95):>8.2f}%  p99={s.quantile(0.99):>8.2f}%")
            else:
                print(f"  {label:<40} mean=${s.mean():>10,.0f}  median=${s.median():>10,.0f}  p95=${s.quantile(0.95):>10,.0f}")

        _row("Position deployed ($)",         "position_cost")
        _row("2-min liquidity window ($)",     "v_2min")
        _row("6-min liquidity window ($)",     "v_6min")
        _row("10-min liquidity window ($)",    "v_10min")
        _row("V_eff_adj (binding cap, $)",     "v_eff_adj")
        _row("Cum dollar vol at entry ($)",    "cum_dvol_at_entry")
        _row("Participation vs V_eff (%)",     "part_veff_pct",  pct=True)
        _row("Participation vs cum dvol (%)",  "part_cum_pct",   pct=True)

        capped = int(hv["vol_capped"].sum())
        print(f"  {'Vol-capped trades':<40} {capped}  ({100*capped/len(hv):.1f}%)")

        # Top 15 largest deployments
        top15 = hv.nlargest(15, "position_cost")
        print(f"\n  TOP 15 LARGEST POSITIONS (by $ deployed)")
        hdr = (f"  {'Date':>10}  {'Ticker':>7}  {'S':>2}  {'Deployed':>9}  "
               f"{'V_2min':>9}  {'V_6min':>9}  {'V_10min':>9}  {'V_eff':>9}  "
               f"{'CumDVol':>12}  {'Part%':>5}  {'CumPart%':>7}  {'PnL':>8}  Cap")
        print(hdr)
        print(f"  {'-'*120}")
        for _, r in top15.iterrows():
            cap = "Y" if r["vol_capped"] else "N"
            print(f"  {r['date']:>10}  {r['ticker']:>7}  {r['strategy']:>2}  "
                  f"${r['position_cost']:>8,.0f}  "
                  f"${r['v_2min']:>8,.0f}  ${r['v_6min']:>8,.0f}  ${r['v_10min']:>8,.0f}  "
                  f"${r['v_eff_adj']:>8,.0f}  ${r['cum_dvol_at_entry']:>11,.0f}  "
                  f"{r['part_veff_pct']:>4.1f}%  {r['part_cum_pct']:>6.2f}%  "
                  f"${r['pnl']:>7,.0f}  {cap}")

        # Highest participation trades (most liquidity stress)
        print(f"\n  TOP 10 HIGHEST PARTICIPATION vs V_EFF (most liquidity stress)")
        top_part = hv.nlargest(10, "part_veff_pct")
        print(f"  {'Date':>10}  {'Ticker':>7}  {'S':>2}  {'Deployed':>9}  {'V_eff':>9}  {'Part%':>6}  {'PnL':>8}  Cap")
        print(f"  {'-'*70}")
        for _, r in top_part.iterrows():
            cap = "Y" if r["vol_capped"] else "N"
            print(f"  {r['date']:>10}  {r['ticker']:>7}  {r['strategy']:>2}  "
                  f"${r['position_cost']:>8,.0f}  ${r['v_eff_adj']:>8,.0f}  "
                  f"{r['part_veff_pct']:>5.1f}%  ${r['pnl']:>7,.0f}  {cap}")

    # Daily breakdown
    print(f"\n  DAILY PnL BREAKDOWN")
    print(f"  {'Date':>10}  {'PnL':>10}  {'Trd':>3}  {'W/L':>5}  {'Equity':>10}  {'DD%':>6}")
    print(f"  {'-'*57}")
    equity_arr = ddf["equity_end"].values
    pk = np.maximum.accumulate(equity_arr)
    dd_pcts = (pk - equity_arr) / pk * 100
    for i, (_, r) in enumerate(ddf.iterrows()):
        sign = "+" if r["pnl"] >= 0 else "-"
        wl   = f"{int(r['wins'])}W/{int(r['losses'])}L"
        dd_s = f"{dd_pcts[i]:.1f}%" if dd_pcts[i] > 0.05 else "    -"
        print(f"  {r['date']:>10}  {sign}${abs(r['pnl']):>9,.2f}  "
              f"{int(r['trades']):>3}  {wl:>5}  ${r['equity_end']:>9,.2f}  {dd_s:>6}")

    print(f"\n  {'='*76}")
    print(f"  TOTAL ${total_pnl:,.2f} ({total_return:.1f}%) | {total_days} days | {total_trades} trades | "
          f"{green_days}G {red_days}R | max DD ${max_dd_abs:,.2f} ({max_dd_pct:.1f}%)")
    print(f"  {'='*76}")


if __name__ == "__main__":
    trades, daily = run_backtest()
    print_report(trades, daily)
