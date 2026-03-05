"""
Analyze dollar volume of stocks we actually entered in the backtest.
Shows how realistic our position sizes would be relative to available liquidity.
"""
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from zoneinfo import ZoneInfo

from test_full import (
    simulate_day, STARTING_CASH, SLIPPAGE_PCT, SPY_SMA_PERIOD,
    MARGIN_THRESHOLD, ADAPTIVE_SIZING, SIZING_TIERS, _get_trade_pct,
)
from optimize import load_all_picks as _load_picks

ET_TZ = ZoneInfo("America/New_York")


def analyze_volume(data_dirs):
    print(f"Loading picks from: {data_dirs}")
    daily_picks = _load_picks(data_dirs)
    dates = sorted(daily_picks.keys())
    print(f"  {len(dates)} trading days: {dates[0]} to {dates[-1]}")

    # Load SPY for regime filter
    spy_df = None
    try:
        spy_df = pd.read_csv("regime_data/spy_daily.csv", index_col=0, parse_dates=True)
    except Exception:
        pass

    cash = STARTING_CASH
    all_entries = []

    for date_str in dates:
        picks = daily_picks[date_str]

        # Regime filter
        if spy_df is not None and SPY_SMA_PERIOD > 0:
            try:
                dt = pd.Timestamp(date_str)
                if spy_df.index.tz:
                    dt = dt.tz_localize(spy_df.index.tz)
                mask = spy_df.index <= dt
                if mask.any():
                    recent = spy_df.loc[mask].tail(SPY_SMA_PERIOD + 1)
                    if len(recent) >= SPY_SMA_PERIOD:
                        sma = recent["Close"].rolling(SPY_SMA_PERIOD).mean().iloc[-1]
                        spy_close = recent["Close"].iloc[-1]
                        if spy_close < sma:
                            continue
            except Exception:
                pass

        cash_account = cash < MARGIN_THRESHOLD
        states, cash = simulate_day(picks, cash, cash_account=cash_account)

        for st in states:
            if st["entry_price"] is None:
                continue

            # Get market hour candles
            mh = st["mh"]
            entry_time = st["entry_time"]

            # Calculate volumes
            pm_vol_shares = st.get("pm_volume", 0)
            pm_dollar_vol = pm_vol_shares * st["entry_price"]  # rough estimate

            # Full day market-hours volume
            day_vol_shares = mh["Volume"].sum()
            day_vwap = (mh["Close"] * mh["Volume"]).sum() / day_vol_shares if day_vol_shares > 0 else st["entry_price"]
            day_dollar_vol = day_vol_shares * day_vwap

            # Volume up to entry time
            pre_entry = mh.loc[mh.index <= entry_time]
            vol_to_entry = pre_entry["Volume"].sum() if len(pre_entry) > 0 else 0
            avg_price_to_entry = pre_entry["Close"].mean() if len(pre_entry) > 0 else st["entry_price"]
            dollar_vol_to_entry = vol_to_entry * avg_price_to_entry

            # Our position size
            position_cost = st["position_cost"]

            # What % of volume would our order be
            pct_of_day = (position_cost / day_dollar_vol * 100) if day_dollar_vol > 0 else 999
            pct_of_pre_entry = (position_cost / dollar_vol_to_entry * 100) if dollar_vol_to_entry > 0 else 999

            month = date_str[:7]

            all_entries.append({
                "date": date_str,
                "month": month,
                "ticker": st["ticker"],
                "entry_price": st["entry_price"],
                "position_cost": position_cost,
                "pm_vol_shares": pm_vol_shares,
                "pm_dollar_vol": pm_dollar_vol,
                "day_vol_shares": int(day_vol_shares),
                "day_dollar_vol": day_dollar_vol,
                "vol_to_entry_shares": int(vol_to_entry),
                "dollar_vol_to_entry": dollar_vol_to_entry,
                "pct_of_day_vol": pct_of_day,
                "pct_of_pre_entry_vol": pct_of_pre_entry,
                "pnl": st["pnl"],
                "gap_pct": st["gap_pct"],
                "cash_at_entry": cash,
            })

    df = pd.DataFrame(all_entries)

    print(f"\n{'='*80}")
    print(f"  VOLUME ANALYSIS — {len(df)} entries across {df['month'].nunique()} months")
    print(f"{'='*80}")

    # Overall stats
    print(f"\n  OVERALL AVERAGES:")
    print(f"  {'Metric':<35} {'Mean':>12} {'Median':>12} {'Min':>12} {'Max':>12}")
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for label, col in [
        ("Position size ($)", "position_cost"),
        ("PM dollar volume ($)", "pm_dollar_vol"),
        ("Full-day dollar volume ($)", "day_dollar_vol"),
        ("Vol to entry - dollar ($)", "dollar_vol_to_entry"),
        ("Our order as % of day vol", "pct_of_day_vol"),
        ("Our order as % of pre-entry vol", "pct_of_pre_entry_vol"),
    ]:
        vals = df[col]
        print(f"  {label:<35} {vals.mean():>12,.1f} {vals.median():>12,.1f} "
              f"{vals.min():>12,.1f} {vals.max():>12,.1f}")

    # Monthly breakdown
    print(f"\n  MONTHLY BREAKDOWN:")
    print(f"  {'Month':<10} {'Entries':>7} {'Avg Pos$':>10} {'Avg Day$Vol':>14} "
          f"{'Avg %DayVol':>12} {'Med %DayVol':>12} {'Max %DayVol':>12}")
    print(f"  {'-'*10} {'-'*7} {'-'*10} {'-'*14} {'-'*12} {'-'*12} {'-'*12}")

    for month, grp in df.groupby("month"):
        print(f"  {month:<10} {len(grp):>7} ${grp['position_cost'].mean():>9,.0f} "
              f"${grp['day_dollar_vol'].mean():>13,.0f} "
              f"{grp['pct_of_day_vol'].mean():>11.1f}% "
              f"{grp['pct_of_day_vol'].median():>11.1f}% "
              f"{grp['pct_of_day_vol'].max():>11.1f}%")

    # Distribution of % of day volume
    print(f"\n  POSITION AS % OF DAY VOLUME — DISTRIBUTION:")
    brackets = [(0, 1), (1, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, 9999)]
    for lo, hi in brackets:
        count = ((df["pct_of_day_vol"] >= lo) & (df["pct_of_day_vol"] < hi)).sum()
        pct = count / len(df) * 100
        label = f"{lo}-{hi}%" if hi < 9999 else f"{lo}%+"
        bar = "#" * int(pct / 2)
        print(f"    {label:<10} {count:>4} ({pct:>5.1f}%)  {bar}")

    # Worst offenders — trades where we'd be >10% of day volume
    big = df[df["pct_of_day_vol"] > 10].sort_values("pct_of_day_vol", ascending=False)
    if len(big) > 0:
        print(f"\n  TRADES WHERE POSITION > 10% OF DAY VOLUME ({len(big)} trades):")
        print(f"  {'Date':<12} {'Ticker':<8} {'Pos$':>10} {'DayVol$':>14} "
              f"{'%DayVol':>8} {'Entry$':>8} {'P&L':>10}")
        print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*14} {'-'*8} {'-'*8} {'-'*10}")
        for _, row in big.head(20).iterrows():
            print(f"  {row['date']:<12} {row['ticker']:<8} ${row['position_cost']:>9,.0f} "
                  f"${row['day_dollar_vol']:>13,.0f} "
                  f"{row['pct_of_day_vol']:>7.1f}% "
                  f"${row['entry_price']:>7.2f} "
                  f"${row['pnl']:>+9,.2f}")

    # What if we capped at $25K per trade?
    print(f"\n  WHAT-IF: CAP AT $25K PER TRADE")
    capped = df.copy()
    capped["capped_pos"] = capped["position_cost"].clip(upper=25000)
    capped["capped_pct_day"] = capped["capped_pos"] / capped["day_dollar_vol"] * 100
    still_big = (capped["capped_pct_day"] > 10).sum()
    print(f"  Trades still >10% of day volume: {still_big}/{len(capped)}")
    print(f"  Avg position: ${capped['capped_pos'].mean():,.0f} (was ${df['position_cost'].mean():,.0f})")

    # What if we capped at $50K per trade?
    print(f"\n  WHAT-IF: CAP AT $50K PER TRADE")
    capped50 = df.copy()
    capped50["capped_pos"] = capped50["position_cost"].clip(upper=50000)
    capped50["capped_pct_day"] = capped50["capped_pos"] / capped50["day_dollar_vol"] * 100
    still_big50 = (capped50["capped_pct_day"] > 10).sum()
    print(f"  Trades still >10% of day volume: {still_big50}/{len(capped50)}")
    print(f"  Avg position: ${capped50['capped_pos'].mean():,.0f} (was ${df['position_cost'].mean():,.0f})")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    dirs = sys.argv[1:] if len(sys.argv) > 1 else ["stored_data", "stored_data_oos"]
    analyze_volume(dirs)
