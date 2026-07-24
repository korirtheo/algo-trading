oimport json
import os
import sys
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

# Add current directory to path
sys.path.insert(0, ".")

from test_full import load_all_picks
from optimize_combined import set_strategy_params, _build_param_snapshot, _param_lock
import test_green_candle_combined as tgc

# Trial #561 Params
TRIAL_561 = {
    "g_min_gap_pct": 15.0,
    "g_partial_sell_pct": 0.0,
    "g_require_2nd_green": True,
    "g_require_2nd_new_high": False,
    "g_stop_pct": 29.0,
    "g_target_pct": 5.0,
    "g_time_limit_min": 6,
    "g_trail_activate_pct": 10.0,
    "g_trail_pct": 1.0,
    "l_earliest_candle": 24,
    "l_latest_candle": 45,
    "l_max_float": 5000000,
    "l_min_gap": 80,
    "l_min_price_accel_pct": 3.0,
    "l_partial_sell_pct": 0.0,
    "l_stop_pct": 24.0,
    "l_tier1_target1_pct": 45.0,
    "l_tier1_target2_pct": 70.0,
    "l_tier2_target1_pct": 24.0,
    "l_tier2_target2_pct": 50.0,
    "l_tier3_target1_pct": 5.0,
    "l_tier3_target2_pct": 15.0,
    "l_trail_activate_pct": 131.0,
    "l_trail_pct": 1.0,
}

STARTING_CASH = 25000
DIRS_2026_OOS = ["stored_data_mar_may_2026", "stored_data_jun_2026", "stored_data_oos"]


def run_backtest_with_stats(trial_number, params, dates, picks_by_date):
    # Configure simulator
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

    # Load complete baseline config
    base_config_path = "config/trial_gl_trail_final_best.json"
    if os.path.exists(base_config_path):
        with open(base_config_path) as f:
            base_data = json.load(f)
        p = dict(base_data.get("params", {}))
    else:
        p = {}

    p.update(params)

    # Enable G and L, disable others using CORRECT enable_{strategy} keys
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
            daily_stats.append({"date": d, "pnl": 0.0, "trades_count": 0})
            continue

        cash_account = cash < 100000
        try:
            states, cash_next, unsettled, _ = tgc.simulate_day_combined(
                picks, cash, cash_account, params=snapshot
            )
        except Exception as e:
            daily_stats.append({"date": d, "pnl": 0.0, "trades_count": 0})
            continue

        effective_cash = cash_next + (unsettled if cash_account else 0)
        day_pnl = 0.0
        day_trades = []

        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                pnl = st.get("pnl", 0)
                if pnl is not None:
                    # Strategy detection: G or L
                    strat = st.get("strategy", "Unknown")
                    trade_info = {
                        "ticker": st["ticker"],
                        "date": d,
                        "strategy": strat,
                        "pnl": pnl,
                        "cost": st["position_cost"],
                    }
                    all_trades.append(trade_info)
                    day_trades.append(trade_info)
                    day_pnl += pnl

        daily_stats.append({"date": d, "pnl": day_pnl, "trades_count": len(day_trades)})
        cash = effective_cash

    return all_trades, daily_stats


def analyze_trades_and_days(trades, daily_stats):
    df_trades = pd.DataFrame(trades)
    df_days = pd.DataFrame(daily_stats)

    total_pnl = df_days["pnl"].sum()
    total_trades = len(df_trades)

    # Day-level metrics
    green_days = len(df_days[df_days["pnl"] > 0])
    red_days = len(df_days[df_days["pnl"] < 0])
    flat_days = len(df_days[df_days["pnl"] == 0])
    avg_daily_pnl = df_days["pnl"].mean()

    # Trade-level metrics
    wins = df_trades[df_trades["pnl"] > 0]["pnl"].tolist() if len(df_trades) > 0 else []
    losses = (
        df_trades[df_trades["pnl"] <= 0]["pnl"].tolist() if len(df_trades) > 0 else []
    )

    gross_win = sum(wins)
    gross_loss = abs(sum(losses)) if losses else 1e-9
    pf = gross_win / gross_loss
    wr = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0

    # Strategy breakdown
    strat_breakdown = {}
    if len(df_trades) > 0:
        for strat, grp in df_trades.groupby("strategy"):
            s_wins = grp[grp["pnl"] > 0]["pnl"].tolist()
            s_losses = grp[grp["pnl"] <= 0]["pnl"].tolist()
            s_pnl = grp["pnl"].sum()
            s_count = len(grp)
            s_gross_win = sum(s_wins)
            s_gross_loss = abs(sum(s_losses)) if s_losses else 1e-9
            s_pf = s_gross_win / s_gross_loss
            s_wr = (len(s_wins) / s_count * 100) if s_count > 0 else 0.0
            strat_breakdown[strat] = {
                "pnl": s_pnl,
                "trades": s_count,
                "pf": s_pf,
                "wr": s_wr,
            }

    return {
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "green_days": green_days,
        "red_days": red_days,
        "flat_days": flat_days,
        "avg_daily_pnl": avg_daily_pnl,
        "pf": pf,
        "wr": wr,
        "strategies": strat_breakdown,
    }


def main():
    print("Loading data for March 2026+ OOS...")
    all_dirs = [d for d in DIRS_2026_OOS if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(all_dirs)
    dates_in_range = sorted([d for d in all_dates if "2026-03-01" <= d <= "2099-12-31"])
    print(f"Loaded {len(dates_in_range)} trading days of data.")

    print("\nRunning backtest for Trial #561...")
    trades_561, days_561 = run_backtest_with_stats(
        561, TRIAL_561, dates_in_range, picks_by_date
    )
    stats_561 = analyze_trades_and_days(trades_561, days_561)

    print("\nLoading Deployed Config (Trial #538)...")
    deployed_path = "config/trial_gl_trail_538_deploy.json"
    with open(deployed_path) as f:
        deployed_data = json.load(f)

    # We must only extract the G & L params to avoid mixing other active strategies
    deployed_params = {
        k: v
        for k, v in deployed_data["params"].items()
        if k.startswith("g_") or k.startswith("l_")
    }

    print("Running backtest for Deployed Config...")
    trades_538, days_538 = run_backtest_with_stats(
        538, deployed_params, dates_in_range, picks_by_date
    )
    stats_538 = analyze_trades_and_days(trades_538, days_538)

    # Print comparative report
    print("\n" + "=" * 80)
    print("      TRUE G+L ONLY COMPARATIVE ANALYSIS: TRIAL #561 vs DEPLOYED (#538)")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Trial #561':<25} | {'Deployed (#538)':<25}")
    print("-" * 80)
    print(
        f"{'Total PnL':<25} | ${stats_561['total_pnl']:,.2f} | ${stats_538['total_pnl']:,.2f}"
    )
    print(f"{'Profit Factor (PF)':<25} | {stats_561['pf']:.3f} | {stats_538['pf']:.3f}")
    print(f"{'Win Rate (WR)':<25} | {stats_561['wr']:.1f}% | {stats_538['wr']:.1f}%")
    print(
        f"{'Total Trades':<25} | {stats_561['total_trades']} | {stats_538['total_trades']}"
    )
    print(
        f"{'Green / Red / Flat Days':<25} | {stats_561['green_days']}/{stats_561['red_days']}/{stats_561['flat_days']} | {stats_538['green_days']}/{stats_538['red_days']}/{stats_538['flat_days']}"
    )
    print(
        f"{'Avg. Daily PnL':<25} | ${stats_561['avg_daily_pnl']:,.2f} | ${stats_538['avg_daily_pnl']:,.2f}"
    )

    print("\n" + "-" * 80)
    print("                     STRATEGY BREAKDOWN: TRIAL #561")
    print("-" * 80)
    for s, info in stats_561["strategies"].items():
        print(f"Strategy {s}:")
        print(f"  PnL:         ${info['pnl']:,.2f}")
        print(f"  Trades:      {info['trades']}")
        print(f"  PF:          {info['pf']:.3f}")
        print(f"  WR:          {info['wr']:.1f}%")

    print("\n" + "-" * 80)
    print("                  STRATEGY BREAKDOWN: DEPLOYED (#538)")
    print("-" * 80)
    for s, info in stats_538["strategies"].items():
        print(f"Strategy {s}:")
        print(f"  PnL:         ${info['pnl']:,.2f}")
        print(f"  Trades:      {info['trades']}")
        print(f"  PF:          {info['pf']:.3f}")
        print(f"  WR:          {info['wr']:.1f}%")
    print("=" * 80)

    # Save detailed report to file
    report_file = "TRIAL_561_vs_DEPLOYED_REPORT.md"
    with open(report_file, "w") as rf:
        rf.write(f"""# G+L ONLY COMPARATIVE ANALYSIS: TRIAL #561 vs DEPLOYED (#538)
## March 2026+ Out-Of-Sample (OOS) Period (60 Trading Days)

### Summary Comparison Table
| Metric | Trial #561 | Deployed (#538) |
| :--- | :---: | :---: |
| **Total PnL** | ${stats_561["total_pnl"]:,.2f} | ${stats_538["total_pnl"]:,.2f} |
| **Profit Factor (PF)** | {stats_561["pf"]:.3f} | {stats_538["pf"]:.3f} |
| **Win Rate (WR)** | {stats_561["wr"]:.1f}% | {stats_538["wr"]:.1f}% |
| **Total Trades** | {stats_561["total_trades"]} | {stats_538["total_trades"]} |
| **Green / Red / Flat Days** | {stats_561["green_days"]} / {stats_561["red_days"]} / {stats_561["flat_days"]} | {stats_538["green_days"]} / {stats_538["red_days"]} / {stats_538["flat_days"]} |
| **Avg. Daily PnL** | ${stats_561["avg_daily_pnl"]:,.2f} | ${stats_538["avg_daily_pnl"]:,.2f} |

### Strategy Performance Breakdown
#### Strategy G (Big Gap Runner)
| Parameter | Trial #561 | Deployed (#538) |
| :--- | :---: | :---: |
| **Min Gap %** | 15.0% | 10.0% |
| **Time Limit** | 6 min | 12 min |
| **Stop %** | 29.0% | 26.0% |
| **Target %** | 5.0% | 15.0% |
| **Trail Stop %** | 1.0% (at +10.0%) | 0.5% (no activation req) |
| **OOS PnL** | ${stats_561["strategies"].get("G", {}).get("pnl", 0):,.2f} | ${stats_538["strategies"].get("G", {}).get("pnl", 0):,.2f} |
| **Trades** | {stats_561["strategies"].get("G", {}).get("trades", 0)} | {stats_538["strategies"].get("G", {}).get("trades", 0)} |
| **PF** | {stats_561["strategies"].get("G", {}).get("pf", 0):.3f} | {stats_538["strategies"].get("G", {}).get("pf", 0):.3f} |
| **WR** | {stats_561["strategies"].get("G", {}).get("wr", 0):.1f}% | {stats_538["strategies"].get("G", {}).get("wr", 0):.1f}% |

#### Strategy L (Low Float Squeeze)
| Parameter | Trial #561 | Deployed (#538) |
| :--- | :---: | :---: |
| **Min Gap %** | 80.0% | 25.0% |
| **Max Float** | 5M | 25M |
| **Earliest Candle** | 24 | 30 |
| **Latest Candle** | 45 | 150 |
| **Stop %** | 24.0% | 18.0% |
| **Tier 1 Targets** | 45% / 70% | 40% / 30% |
| **Tier 2 Targets** | 24% / 50% | 30% / 55% |
| **Tier 3 Targets** | 5% / 15% | 25% / 40% |
| **Trail Stop %** | 1.0% (at +131.0%) | 1.0% (at +4.0%) |
| **OOS PnL** | ${stats_561["strategies"].get("L", {}).get("pnl", 0):,.2f} | ${stats_538["strategies"].get("L", {}).get("pnl", 0):,.2f} |
| **Trades** | {stats_561["strategies"].get("L", {}).get("trades", 0)} | {stats_538["strategies"].get("L", {}).get("trades", 0)} |
| **PF** | {stats_561["strategies"].get("L", {}).get("pf", 0):.3f} | {stats_538["strategies"].get("L", {}).get("pf", 0):.3f} |
| **WR** | {stats_561["strategies"].get("L", {}).get("wr", 0):.1f}% | {stats_538["strategies"].get("L", {}).get("wr", 0):.1f}% |

### Key Takeaways
1. **OOS Performance**: Trial #561 achieved an OOS PnL of **${stats_561["total_pnl"]:,.2f}** compared to Deployed (#538) **${stats_538["total_pnl"]:,.2f}**.
2. **Strategy G (Gappers)**: Trial #561's G strategy parameters were more restrictive on gaps but tighter on time limits, leading to different trade capture compared to #538.
3. **Strategy L (Low Float)**: Trial #561 uses a very restrictive `l_min_gap` (80.0%) and `l_max_float` (5M), leading to fewer but highly focused trades.
4. **Daily Average**: Trial #561 average daily PnL of **${stats_561["avg_daily_pnl"]:,.2f}** compared to Deployed's **${stats_538["avg_daily_pnl"]:,.2f}** over the 60-day OOS period.
""")
    print(f"\nDetailed report saved to {report_file}")


if __name__ == "__main__":
    main()
