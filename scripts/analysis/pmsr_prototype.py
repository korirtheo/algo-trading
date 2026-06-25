"""Pre-Market Strong Runner (PMSR) prototype.

Triggers DIFFERENTLY from G:
  - G fires when 2nd candle is green + makes new HOD (continuation pattern)
  - PMSR fires when stock breaks pre-market high within first N candles
    of regular hours (breakout pattern)

These are STRUCTURALLY different signals:
  - G requires a green 2nd candle (means there was a pullback in candle 1)
  - PMSR catches VERTICAL pumps that don't pull back — the candles G misses

Strategy spec:
  Eligibility:
    - gap_pct >= 30%
    - pm_volume >= 1M shares  (real PM engagement)
  Entry:
    - First bar of regular hours that breaks premarket_high
    - Within first 10 candles (20 min)
    - Bar must be green (close > open)
    - Volume confirmation (vol > 1.5x avg)
  Exit:
    - Hard stop: -8% from entry
    - Target: +20%
    - Time stop: 30 min
"""
import json
import os
import sys
import pickle
import numpy as np
from datetime import timedelta

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
            if date not in all_picks:
                all_picks[date] = picks

dates = sorted(all_picks.keys())
print(f"Total dates: {len(dates)}")


def simulate_pmsr_day(picks, cash):
    """Simulate PMSR on one day. One position at a time, like G."""
    PMSR_MIN_GAP = 30.0
    PMSR_MIN_PM_VOL = 1_000_000
    PMSR_MAX_ENTRY_CANDLE = 10
    PMSR_STOP_PCT = 8.0
    PMSR_TARGET_PCT = 20.0
    PMSR_VOL_MULT = 1.5
    PMSR_TIME_LIMIT_MIN = 30

    # Filter eligible
    eligible = [p for p in picks
                if p.get("gap_pct", 0) >= PMSR_MIN_GAP
                and p.get("pm_volume", 0) >= PMSR_MIN_PM_VOL
                and p.get("premarket_high") is not None
                and p.get("market_hour_candles") is not None
                and len(p.get("market_hour_candles", [])) > 5]

    trades = []
    active_position = None
    pmsr_states = {}

    # Build per-ticker timeline of bars + check breakout
    for p in eligible:
        ticker = p["ticker"]
        pmh = p["premarket_high"]
        mh = p["market_hour_candles"]
        if len(mh) < PMSR_MAX_ENTRY_CANDLE:
            continue

        # Find entry candle: first bar where High > pmh, within first N candles, green, vol confirmed
        entry_idx = None
        avg_vol = None
        for i in range(PMSR_MAX_ENTRY_CANDLE):
            bar = mh.iloc[i]
            if bar["High"] > pmh and bar["Close"] > bar["Open"]:
                if i >= 2:
                    avg_vol = mh.iloc[max(0, i-3):i]["Volume"].mean()
                    if avg_vol > 0 and bar["Volume"] < avg_vol * PMSR_VOL_MULT:
                        continue
                entry_idx = i
                break

        if entry_idx is None: continue
        pmsr_states[ticker] = {"entry_idx": entry_idx, "entry_price": mh.iloc[entry_idx]["Close"], "mh": mh, "pmh": pmh}

    # Now simulate trade execution: priority by gap_pct descending
    pmsr_states = dict(sorted(pmsr_states.items(),
                               key=lambda kv: -next((p["gap_pct"] for p in eligible if p["ticker"] == kv[0]), 0)))

    for ticker, st in pmsr_states.items():
        if active_position is not None: continue  # one position at a time
        entry_idx = st["entry_idx"]
        entry_price = st["entry_price"]
        mh = st["mh"]
        # Cash deploy: full cash (matching backtest convention)
        position_cost = cash
        shares = position_cost / entry_price

        # Track exit
        stop_price = entry_price * (1 - 8/100)
        target_price = entry_price * (1 + 20/100)
        exit_at_bar = None
        exit_reason = None
        max_bars = PMSR_TIME_LIMIT_MIN // 2  # 2-min bars

        for j in range(entry_idx + 1, min(len(mh), entry_idx + 1 + max_bars)):
            bar = mh.iloc[j]
            if bar["Low"] <= stop_price:
                exit_at_bar = j; exit_reason = "STOP"; exit_price = stop_price; break
            if bar["High"] >= target_price:
                exit_at_bar = j; exit_reason = "TARGET"; exit_price = target_price; break

        if exit_reason is None:
            # Time-stop at end of allowed bars
            j_end = min(len(mh) - 1, entry_idx + max_bars)
            exit_at_bar = j_end
            exit_reason = "TIME"
            exit_price = mh.iloc[j_end]["Close"]

        pnl = (exit_price - entry_price) * shares - 0.01 * shares  # tiny slippage
        cash = cash + pnl
        active_position = ticker
        trades.append({
            "ticker": ticker, "entry_idx": entry_idx, "entry_price": entry_price,
            "exit_price": exit_price, "exit_reason": exit_reason,
            "pnl": pnl, "ret_pct": (exit_price/entry_price - 1) * 100,
            "gap_pct": next((p["gap_pct"] for p in eligible if p["ticker"] == ticker), 0),
        })
        break  # only one trade per day for now

    return trades, cash


# Run on full backtest
daily_results = {}
for d in dates:
    picks = all_picks.get(d, [])
    if not picks: continue
    trades, _ = simulate_pmsr_day(picks, STARTING_CASH)
    daily_results[d] = trades

# Stats
all_trades = [t for d in dates for t in daily_results.get(d, [])]
n_trades = len(all_trades)
total_pnl = sum(t["pnl"] for t in all_trades)
wins = [t for t in all_trades if t["pnl"] > 0]
losses = [t for t in all_trades if t["pnl"] <= 0]
wr = len(wins)/n_trades*100 if n_trades else 0

print(f"\n=== PMSR (default params) on {len(dates)} days ===")
print(f"  Trades:      {n_trades}")
print(f"  WR:          {wr:.1f}%")
print(f"  Avg ret/trade: {np.mean([t['ret_pct'] for t in all_trades]):.2f}%" if all_trades else "")
print(f"  Total $ (fresh $25K per day): ${total_pnl:,.0f}")
if all_trades:
    print(f"  Avg per-trade $: ${total_pnl/n_trades:+,.0f}")
    print(f"  Best winner: +{max(t['ret_pct'] for t in all_trades):.1f}%")
    print(f"  Worst loser: {min(t['ret_pct'] for t in all_trades):.1f}%")
    by_exit = {}
    for t in all_trades:
        by_exit.setdefault(t["exit_reason"], []).append(t)
    print(f"\n  Exit breakdown:")
    for reason, ts in by_exit.items():
        wn = sum(1 for t in ts if t["pnl"] > 0)
        print(f"    {reason}: n={len(ts)} WR={wn/len(ts)*100:.0f}% avg=${sum(t['pnl'] for t in ts)/len(ts):+,.0f}")

# Compare: how many PMSR trades happened on G's zero-trade days?
import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
with open("config/trial_w13_1202_deploy.json") as f: p_dep = json.load(f)["params"]
with open("config/trial_432_params.json") as f: baseline = json.load(f)
merged = {**baseline, **p_dep}
for s in ALL_STRATS: merged[f"enable_{s}"] = (s in {"g","l"})
set_strategy_params(merged)
tgc.USE_DYNAMIC_SLIPPAGE = True; tgc.USE_MULTIWINDOW_SLIPPAGE = True
tgc.USE_VOLATILITY_ADJUSTMENT = True
tgc.SLIP_IMPACT_K = 3.0; tgc.VOL_CAP_PCT = 5.0
tgc.MAX_2MIN_PARTICIPATION = 0.15; tgc.MAX_REGIME_PARTICIPATION = 0.08
tgc.MARGIN_MULTIPLIER = 1.0; tgc.NEWS_MODULATOR_ENABLED = False

pmsr_only_dates = []
both_dates = []
for d in dates:
    pmsr_trades = daily_results.get(d, [])
    g_picks = all_picks.get(d, [])
    if not g_picks or not pmsr_trades: continue
    try:
        states, _, _, _ = tgc.simulate_day_combined(g_picks, STARTING_CASH, cash_account=True)
        g_trades = [s for s in states if s.get("exit_reason") is not None and s.get("position_cost", 0) > 0]
    except Exception:
        g_trades = []
    if g_trades:
        both_dates.append(d)
    else:
        pmsr_only_dates.append(d)

print(f"\n=== Orthogonality with G+L ===")
print(f"Days PMSR fires:           {sum(1 for d in dates if daily_results.get(d, []))}")
print(f"  ...AND G fires:          {len(both_dates)} (overlap)")
print(f"  ...G is silent (NEW!):   {len(pmsr_only_dates)} (incremental)")

# Save trades for further analysis
with open("results/pmsr_trades.json", "w") as f:
    json.dump([{"date": d, **t} for d in dates for t in daily_results.get(d, [])], f, indent=2, default=str)
print(f"\nSaved: results/pmsr_trades.json")
