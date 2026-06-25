"""Multi-day continuation (MDC).

Hypothesis: stocks that pumped massively yesterday (>=25% intraday gain)
often continue up the next day. Buy them on Day 2 if they're still showing
strength at the open.

Setup spec:
  Eligibility (Day 1 = yesterday):
    - Daily return (close vs open) >= 25%
    - Gapped >= 15% from prior_close that day (was real pump candidate)
    - Closed at least 70% of the way to its daily high (strong close, not faded)
  Entry (Day 2 = today):
    - Ticker found in today's picks pkl (so we have data)
    - Today's open >= yesterday's close (still gapping or unchanged)
    - Entry at first 2-min bar's close after market open
  Exit:
    - Stop: -10% from entry
    - Target: +20%
    - Trail: 5% after +5% activation
    - Time: 60 min from entry
"""
import json
import os
import sys
import pickle
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, '.')

STARTING_CASH = 25_000
DATA_DIRS = ["stored_data_combined", "stored_data", "stored_data_2022", "stored_data_2023",
             "stored_data_mar_may_2026", "stored_data_jun_2026"]

# Load picks
all_picks = {}
for d in DATA_DIRS:
    p = os.path.join(d, "fulltest_picks_gap2_vol250k.pkl")
    if not os.path.exists(p): continue
    with open(p, "rb") as f:
        for date, picks in pickle.load(f).items():
            if date not in all_picks: all_picks[date] = picks

dates = sorted(all_picks.keys())
print(f"Total trading dates: {len(dates)}")

# Build per-date ticker -> pick lookup for fast access
ticker_lookup = {d: {p["ticker"]: p for p in all_picks[d]} for d in dates}

# Params
MDC_MIN_DAY1_RETURN = 25.0   # Day 1 intraday return threshold
MDC_MIN_DAY1_GAP = 15.0      # Day 1 gap threshold (was real pump candidate)
MDC_DAY1_CLOSE_TO_HIGH_PCT = 70.0  # close at least N% of way to daily high
MDC_STOP_PCT = 10.0
MDC_TARGET_PCT = 20.0
MDC_TRAIL_PCT = 5.0
MDC_TRAIL_ACTIVATE_PCT = 5.0
MDC_TIME_LIMIT_MIN = 60


def get_day1_pumpers(date_str):
    """For each pick on date_str, determine if it was a 'monster pump' day:
       - Gap >= 15%
       - Daily return (close - open) / open >= 25%
       - Close at least 70% of the way to the day's high
    Returns: list of ticker symbols that pumped.
    """
    picks = all_picks.get(date_str, [])
    pumpers = []
    for p in picks:
        if p.get("gap_pct", 0) < MDC_MIN_DAY1_GAP: continue
        mh = p.get("market_hour_candles")
        if mh is None or len(mh) < 10: continue
        day_open = float(mh.iloc[0]["Open"])
        day_close = float(mh.iloc[-1]["Close"])
        day_high = float(mh["High"].max())
        if day_open <= 0: continue
        day_ret = (day_close / day_open - 1) * 100
        if day_ret < MDC_MIN_DAY1_RETURN: continue
        # Close strength: how close to high?
        if day_high > day_open:
            close_pct = (day_close - day_open) / (day_high - day_open) * 100
            if close_pct < MDC_DAY1_CLOSE_TO_HIGH_PCT: continue
        pumpers.append({
            "ticker": p["ticker"],
            "day1_date": date_str,
            "day1_open": day_open,
            "day1_close": day_close,
            "day1_high": day_high,
            "day1_gap": p.get("gap_pct", 0),
            "day1_return": day_ret,
        })
    return pumpers


def simulate_mdc_entry(today_pick, day1_info):
    """Simulate Day 2 entry for a multi-day continuation candidate."""
    mh = today_pick.get("market_hour_candles")
    if mh is None or len(mh) < 5: return None

    # Today must open >= yesterday's close (still strong)
    today_open = float(mh.iloc[0]["Open"])
    if today_open < day1_info["day1_close"]:
        return None  # Failed to maintain — skip

    # Enter at end of first 2-min bar
    entry_price = float(mh.iloc[0]["Close"])
    entry_price *= 1.003  # 30 bp slippage
    position_cost = STARTING_CASH
    shares = position_cost / entry_price

    stop_price = entry_price * (1 - MDC_STOP_PCT / 100)
    target_price = entry_price * (1 + MDC_TARGET_PCT / 100)
    trail_activated = False
    trail_high = entry_price

    max_bars = MDC_TIME_LIMIT_MIN // 2
    exit_price = None; exit_reason = None
    for j in range(1, min(len(mh), 1 + max_bars)):
        bar = mh.iloc[j]
        if bar["Low"] <= stop_price:
            exit_price = stop_price * 0.998
            exit_reason = "STOP"; break
        if bar["High"] >= target_price:
            exit_price = target_price * 0.998
            exit_reason = "TARGET"; break
        if bar["High"] > trail_high:
            trail_high = float(bar["High"])
        if not trail_activated and (trail_high - entry_price) / entry_price * 100 >= MDC_TRAIL_ACTIVATE_PCT:
            trail_activated = True
        if trail_activated:
            trail_stop = trail_high * (1 - MDC_TRAIL_PCT / 100)
            if bar["Low"] <= trail_stop:
                exit_price = trail_stop * 0.998
                exit_reason = "TRAIL"; break

    if exit_reason is None:
        j_end = min(len(mh) - 1, max_bars)
        exit_price = float(mh.iloc[j_end]["Close"]) * 0.998
        exit_reason = "TIME"

    pnl = (exit_price - entry_price) * shares
    return {
        "ticker": today_pick["ticker"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "pnl": pnl,
        "ret_pct": (exit_price / entry_price - 1) * 100,
        "day1_ret": day1_info["day1_return"],
        "day1_gap": day1_info["day1_gap"],
        "day2_open_pct": (today_open / day1_info["day1_close"] - 1) * 100,
    }


# Run through dates pairwise
trades_by_date = defaultdict(list)
n_pumpers_total = 0
n_pumpers_with_day2_data = 0
for i, d in enumerate(dates):
    if i == 0: continue
    day1 = dates[i-1]
    pumpers = get_day1_pumpers(day1)
    n_pumpers_total += len(pumpers)
    # For each pumper, check Day 2 (= today's data)
    for pumper in pumpers:
        ticker = pumper["ticker"]
        if ticker not in ticker_lookup.get(d, {}): continue
        n_pumpers_with_day2_data += 1
        today_pick = ticker_lookup[d][ticker]
        trade = simulate_mdc_entry(today_pick, pumper)
        if trade is not None:
            trades_by_date[d].append(trade)

all_trades = [t for d, ts in trades_by_date.items() for t in ts]
n = len(all_trades)
wins = [t for t in all_trades if t["pnl"] > 0]
total_pnl = sum(t["pnl"] for t in all_trades)
print(f"\n=== Multi-day continuation (MDC) ===")
print(f"  Day 1 pumpers detected:      {n_pumpers_total}")
print(f"  Pumpers with Day 2 data:     {n_pumpers_with_day2_data}")
print(f"  Trades fired (passed Day 2 entry filter): {n}")
print(f"  WR:           {len(wins)/n*100 if n else 0:.1f}%")
print(f"  Total $:      ${total_pnl:,.0f}")
if all_trades:
    print(f"  Avg ret/trade: {np.mean([t['ret_pct'] for t in all_trades]):.2f}%")
    print(f"  Avg per-trade: ${total_pnl/n:+,.0f}")
    print(f"  Best winner:  +{max(t['ret_pct'] for t in all_trades):.1f}%")
    print(f"  Worst loser:  {min(t['ret_pct'] for t in all_trades):.1f}%")

    # Exit breakdown
    by_exit = defaultdict(list)
    for t in all_trades: by_exit[t["exit_reason"]].append(t)
    print(f"\n  Exit breakdown:")
    for reason, ts in by_exit.items():
        wn = sum(1 for t in ts if t["pnl"] > 0)
        total = sum(t["pnl"] for t in ts)
        print(f"    {reason}: n={len(ts):>3} WR={wn/len(ts)*100:>4.1f}% avg=${total/len(ts):>+7,.0f}")

# Multi-year breakdown
print(f"\n=== Multi-year ===")
by_year = defaultdict(list)
for d, ts in trades_by_date.items():
    for t in ts: by_year[d[:4]].append(t)
for year in sorted(by_year.keys()):
    ts = by_year[year]
    if not ts: continue
    wins_y = sum(1 for t in ts if t["pnl"] > 0)
    total = sum(t["pnl"] for t in ts)
    print(f"  {year}: n={len(ts):>3} WR={wins_y/len(ts)*100:>4.1f}% total=${total:>+10,.0f}  avg=${total/len(ts):>+8,.0f}")

# Orthogonality with G+L: how many MDC trades on G-silent days?
import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
with open("config/trial_w13_1202_deploy.json") as f: p_dep = json.load(f)["params"]
with open("config/trial_432_params.json") as f: baseline = json.load(f)
merged = {**baseline, **p_dep}
for s in ALL_STRATS: merged[f"enable_{s}"] = (s in {"g","l"})
set_strategy_params(merged)
tgc.USE_DYNAMIC_SLIPPAGE = True; tgc.USE_MULTIWINDOW_SLIPPAGE = True
tgc.USE_VOLATILITY_ADJUSTMENT = True; tgc.SLIP_IMPACT_K = 3.0; tgc.VOL_CAP_PCT = 5.0
tgc.MAX_2MIN_PARTICIPATION = 0.15; tgc.MAX_REGIME_PARTICIPATION = 0.08
tgc.MARGIN_MULTIPLIER = 1.0; tgc.NEWS_MODULATOR_ENABLED = False

g_silent_days = set(); g_fires_days = set()
for d in dates:
    picks = all_picks[d]
    if not picks: continue
    try:
        states, _, _, _ = tgc.simulate_day_combined(picks, 25000, cash_account=True)
        has_g = any(s.get("exit_reason") is not None and s.get("position_cost", 0) > 0 for s in states)
    except Exception: has_g = False
    (g_fires_days if has_g else g_silent_days).add(d)

inc = [t for d, ts in trades_by_date.items() if d in g_silent_days for t in ts]
ovr = [t for d, ts in trades_by_date.items() if d in g_fires_days for t in ts]
print(f"\n=== Orthogonality with G+L ===")
print(f"MDC on G-silent days (INCREMENTAL): n={len(inc)} "
      f"WR={sum(1 for t in inc if t['pnl']>0)/len(inc)*100 if inc else 0:.1f}% "
      f"total=${sum(t['pnl'] for t in inc):,.0f}")
print(f"MDC on G-fires days (overlap):     n={len(ovr)} "
      f"WR={sum(1 for t in ovr if t['pnl']>0)/len(ovr)*100 if ovr else 0:.1f}% "
      f"total=${sum(t['pnl'] for t in ovr):,.0f}")

# Mar-Jun 2026 blind
mar_jun = [t for d, ts in trades_by_date.items() if "2026-03-01" <= d <= "2026-12-31" for t in ts]
print(f"\n=== Mar-Jun 2026 blind OOS ===")
print(f"MDC: n={len(mar_jun)} "
      f"WR={sum(1 for t in mar_jun if t['pnl']>0)/len(mar_jun)*100 if mar_jun else 0:.1f}% "
      f"total=${sum(t['pnl'] for t in mar_jun):,.0f}")

with open("results/mdc_trades.json", "w") as f:
    json.dump([{"date": d, **t} for d, ts in trades_by_date.items() for t in ts], f, indent=2, default=str)
print(f"\nSaved: results/mdc_trades.json")
