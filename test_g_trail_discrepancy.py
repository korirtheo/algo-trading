import json
import os
import sys
import numpy as np
import pandas as pd

# Add current directory to path
sys.path.insert(0, ".")

from test_full import load_all_picks
from optimize_combined import set_strategy_params, _build_param_snapshot, _param_lock
import test_green_candle_combined as tgc

STARTING_CASH = 25000
DIRS_2026_OOS = ["stored_data_mar_may_2026", "stored_data_jun_2026", "stored_data_oos"]


def run_backtest(params, dates, picks_by_date):
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False
    tgc.MIN_PRICE = 0.0
    tgc.MAX_MODELED_SLIP_BP = 0.0
    tgc.MAX_CUM_DVOL_AT_ENTRY_M = 0.0
    tgc.MIN_ATR_PCT = 0.0
    tgc.MIN_FAVORABILITY_THRESHOLD = 0.0
    tgc.NEWS_FILTER_ENABLED = False

    base_config_path = "config/trial_gl_trail_final_best.json"
    if os.path.exists(base_config_path):
        with open(base_config_path) as f:
            base_data = json.load(f)
        p = dict(base_data.get("params", {}))
    else:
        p = {}

    p.update(params)
    p["enable_g"] = True
    p["enable_l"] = True
    for prefix in "hafdvmrpwobkcsexijn":
        p[f"enable_{prefix}"] = False

    with _param_lock:
        set_strategy_params(p)
        snapshot = _build_param_snapshot()

    cash = float(STARTING_CASH)
    all_trades = []
    daily_stats = []

    for d in dates:
        picks = picks_by_date.get(d, [])
        if not picks:
            daily_stats.append({"date": d, "pnl": 0.0})
            continue

        cash_account = cash < 100000
        try:
            states, cash_next, unsettled, _ = tgc.simulate_day_combined(
                picks, cash, cash_account, params=snapshot
            )
        except Exception:
            daily_stats.append({"date": d, "pnl": 0.0})
            continue

        effective_cash = cash_next + (unsettled if cash_account else 0)
        day_pnl = 0.0
        day_trades = []

        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                pnl = st.get("pnl", 0)
                if pnl is not None:
                    strat = st.get("strategy", "Unknown")
                    trade_info = {
                        "ticker": st["ticker"],
                        "date": d,
                        "strategy": strat,
                        "pnl": pnl,
                        "cost": st["position_cost"],
                        "exit_reason": st.get("exit_reason"),
                    }
                    all_trades.append(trade_info)
                    day_trades.append(trade_info)
                    day_pnl += pnl

        daily_stats.append({"date": d, "pnl": day_pnl, "trades_count": len(day_trades)})
        cash = effective_cash

    return all_trades, daily_stats


def analyze_results(trades, daily_stats):
    df_trades = pd.DataFrame(trades)
    df_days = pd.DataFrame(daily_stats)
    total_pnl = df_days["pnl"].sum()
    total_trades = len(df_trades)
    wins = df_trades[df_trades["pnl"] > 0]["pnl"].tolist() if len(df_trades) > 0 else []
    losses = (
        df_trades[df_trades["pnl"] <= 0]["pnl"].tolist() if len(df_trades) > 0 else []
    )
    gross_win = sum(wins)
    gross_loss = abs(sum(losses)) if losses else 1e-9
    pf = gross_win / gross_loss
    wr = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0

    trail_exits = (
        len(df_trades[df_trades["exit_reason"] == "TRAIL"]) if len(df_trades) > 0 else 0
    )
    target_exits = (
        len(df_trades[df_trades["exit_reason"] == "TARGET"])
        if len(df_trades) > 0
        else 0
    )
    stop_exits = (
        len(df_trades[df_trades["exit_reason"] == "STOP"]) if len(df_trades) > 0 else 0
    )
    time_exits = (
        len(df_trades[df_trades["exit_reason"] == "TIME_STOP"])
        if len(df_trades) > 0
        else 0
    )

    return {
        "pnl": total_pnl,
        "trades": total_trades,
        "pf": pf,
        "wr": wr,
        "trail_exits": trail_exits,
        "target_exits": target_exits,
        "stop_exits": stop_exits,
        "time_exits": time_exits,
        "trades_list": trades,
    }


def main():
    print("Loading data for March 2026+ OOS...")
    all_dirs = [d for d in DIRS_2026_OOS if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(all_dirs)
    dates_in_range = sorted([d for d in all_dates if "2026-03-01" <= d <= "2099-12-31"])

    deployed_path = "config/trial_gl_trail_538_deploy.json"
    with open(deployed_path) as f:
        deployed_data = json.load(f)

    deployed_params = {
        k: v
        for k, v in deployed_data["params"].items()
        if k.startswith("g_") or k.startswith("l_")
    }

    # Condition 1: Trailing stop inactive (g_trail_pct = 0.0)
    print("\n--- Running with Trailing Stop INACTIVE (g_trail_pct = 0.0%) ---")
    deployed_params["g_trail_pct"] = 0.0
    trades_00, days_00 = run_backtest(deployed_params, dates_in_range, picks_by_date)
    stats_00 = analyze_results(trades_00, days_00)

    # Condition 2: Trailing stop active at 0.5% (default)
    print("\n--- Running with Trailing Stop Active (g_trail_pct = 0.5%) ---")
    deployed_params["g_trail_pct"] = 0.5
    trades_05, days_05 = run_backtest(deployed_params, dates_in_range, picks_by_date)
    stats_05 = analyze_results(trades_05, days_05)

    # Condition 3: Trailing stop active at 1.0%
    print("\n--- Running with Trailing Stop Active (g_trail_pct = 1.0%) ---")
    deployed_params["g_trail_pct"] = 1.0
    trades_10, days_10 = run_backtest(deployed_params, dates_in_range, picks_by_date)
    stats_10 = analyze_results(trades_10, days_10)

    # Condition 4: Trailing stop active at 1.5%
    print("\n--- Running with Trailing Stop Active (g_trail_pct = 1.5%) ---")
    deployed_params["g_trail_pct"] = 1.5
    trades_15, days_15 = run_backtest(deployed_params, dates_in_range, picks_by_date)
    stats_15 = analyze_results(trades_15, days_15)

    print("\n" + "=" * 110)
    print(
        "                      SENSITIVITY ANALYSIS: STRATEGY G TRAILING STOP SWEEP ON 2026 OOS"
    )
    print("=" * 110)
    print(
        f"{'Metric':<25} | {'Inactive (0.0%)':<18} | {'Active (0.5%)':<18} | {'Active (1.0%)':<18} | {'Active (1.5%)':<18}"
    )
    print("-" * 110)
    print(
        f"{'Total PnL':<25} | ${stats_00['pnl']:,.2f} | ${stats_05['pnl']:,.2f} | ${stats_10['pnl']:,.2f} | ${stats_15['pnl']:,.2f}"
    )
    print(
        f"{'Profit Factor (PF)':<25} | {stats_00['pf']:.3f} | {stats_05['pf']:.3f} | {stats_10['pf']:.3f} | {stats_15['pf']:.3f}"
    )
    print(
        f"{'Win Rate (WR)':<25} | {stats_00['wr']:.1f}% | {stats_05['wr']:.1f}% | {stats_10['wr']:.1f}% | {stats_15['wr']:.1f}%"
    )
    print(
        f"{'Total Trades':<25} | {stats_00['trades']} | {stats_05['trades']} | {stats_10['trades']} | {stats_15['trades']}"
    )

    print("\nStrategy G Exit Reason Count comparison:")
    for s, name in [
        (stats_00, "0.0%"),
        (stats_05, "0.5%"),
        (stats_10, "1.0%"),
        (stats_15, "1.5%"),
    ]:
        df = pd.DataFrame(s["trades_list"])
        g_df = df[df["strategy"] == "G"] if len(df) > 0 else pd.DataFrame()
        trail = len(g_df[g_df["exit_reason"] == "TRAIL"]) if len(g_df) > 0 else 0
        target = len(g_df[g_df["exit_reason"] == "TARGET"]) if len(g_df) > 0 else 0
        stop = len(g_df[g_df["exit_reason"] == "STOP"]) if len(g_df) > 0 else 0
        t_stop = len(g_df[g_df["exit_reason"] == "TIME_STOP"]) if len(g_df) > 0 else 0
        print(
            f"  g_trail_pct={name:<5} | Trail Exits: {trail:<3} | Target Exits: {target:<3} | Stop Exits: {stop:<3} | Time Stop Exits: {t_stop:<3}"
        )
    print("=" * 110)


if __name__ == "__main__":
    main()
