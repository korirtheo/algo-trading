"""
Analyze winners vs losers from 2026 OOS backtest with Trial #538 config.
Find patterns that predict which trades win and which lose.
"""
import sys
import os
import json
sys.path.insert(0, ".")

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks
from collections import defaultdict

def run_analysis():
    # Load Trial #538 config and apply ALL params (including enable/disable)
    with open("config/trial_gl_trail_538_deploy.json") as f:
        data = json.load(f)
    params = data["params"]

    # This sets all strategy globals AND disables strategies via MIN_GAP_PCT = 9999
    set_strategy_params(params)

    # Snapshot original values for restoration later
    originals = {k: getattr(tgc, k) for k in dir(tgc) if k.isupper() and not k.startswith("_")}

    data_dirs = ["stored_data_mar_may_2026", "stored_data_2026"]
    data_dir = None
    for d in data_dirs:
        if os.path.exists(d):
            data_dir = d
            break

    all_dates, daily_picks = load_all_picks([data_dir])

    winners = []
    losers = []

    for date in all_dates:
        picks = daily_picks.get(date, [])
        if not picks:
            continue

        states, cash, unsettled, log = tgc.simulate_day_combined(
            picks=picks, cash=25000, cash_account=False, is_live=False, params=None
        )

        for st in states:
            if st.get("exit_time") and st.get("pnl") is not None:
                # Build enriched trade record from state + pick data
                trade = {
                    "strategy": st.get("strategy"),
                    "ticker": st.get("ticker"),
                    "date": date,
                    "entry_price": st.get("entry_price_original", st.get("entry_price", 0)),
                    "exit_price": st.get("exit_price", 0),
                    "pnl": st["pnl"],
                    "pnl_pct": st.get("pnl_pct", 0),
                    "reason": st.get("exit_reason", "?"),
                    "shares": st.get("shares", 0),
                    "hold_time_min": None,
                }

                # Calculate hold time
                entry_t = st.get("entry_time")
                exit_t = st.get("exit_time")
                if entry_t and exit_t:
                    try:
                        from datetime import datetime
                        if isinstance(entry_t, str):
                            e1 = datetime.fromisoformat(entry_t.replace("+00:00", ""))
                        else:
                            e1 = entry_t
                        if isinstance(exit_t, str):
                            e2 = datetime.fromisoformat(exit_t.replace("+00:00", ""))
                        else:
                            e2 = exit_t
                        trade["hold_time_min"] = (e2 - e1).total_seconds() / 60.0
                    except:
                        pass

                # Get pick data for gap%, float, entry candle
                for p in picks:
                    if p.get("ticker") == st.get("ticker"):
                        trade["gap_pct"] = p.get("gap_pct", 0)
                        trade["float_shares"] = p.get("float_shares", 0)
                        trade["entry_candle"] = p.get("entry_candle", 0)
                        trade["pm_volume"] = p.get("pm_volume", 0)
                        break
                else:
                    trade["gap_pct"] = 0
                    trade["float_shares"] = 0
                    trade["entry_candle"] = 0
                    trade["pm_volume"] = 0

                if trade["pnl"] > 0:
                    winners.append(trade)
                else:
                    losers.append(trade)

    all_trades = winners + losers

    print("=" * 80)
    print("WINNERS vs LOSERS ANALYSIS — 2026 OOS (Trial #538 Config)")
    print("=" * 80)
    print(f"Total trades: {len(all_trades)} | Winners: {len(winners)} | Losers: {len(losers)}")
    print(f"Win rate: {len(winners)/len(all_trades)*100:.1f}%")
    print(f"Avg winner P&L: ${sum(t['pnl'] for t in winners)/len(winners):,.0f}")
    print(f"Avg loser P&L: ${sum(t['pnl'] for t in losers)/len(losers):,.0f}")
    print(f"Total P&L: ${sum(t['pnl'] for t in all_trades):,.0f}")

    # ---- PATTERN 1: By Strategy ----
    print("\n" + "=" * 80)
    print("PATTERN 1: WIN RATE BY STRATEGY")
    print("=" * 80)
    strat_data = defaultdict(lambda: {"w": 0, "l": 0, "pnl": 0})
    for t in all_trades:
        s = t["strategy"]
        strat_data[s]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            strat_data[s]["w"] += 1
        else:
            strat_data[s]["l"] += 1

    for s in sorted(strat_data.keys()):
        d = strat_data[s]
        total = d["w"] + d["l"]
        wr = d["w"] / total * 100 if total else 0
        avg_pnl = d["pnl"] / total if total else 0
        print(f"  {s}: {total:4d} trades | WR: {wr:5.1f}% | Avg P&L: ${avg_pnl:8,.0f} | Total: ${d['pnl']:10,.0f}")

    # ---- PATTERN 2: By Exit Reason ----
    print("\n" + "=" * 80)
    print("PATTERN 2: WIN RATE BY EXIT REASON")
    print("=" * 80)
    reason_data = defaultdict(lambda: {"w": 0, "l": 0, "pnl": 0})
    for t in all_trades:
        r = t["reason"]
        reason_data[r]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            reason_data[r]["w"] += 1
        else:
            reason_data[r]["l"] += 1

    for r in sorted(reason_data.keys(), key=lambda x: reason_data[x]["pnl"], reverse=True):
        d = reason_data[r]
        total = d["w"] + d["l"]
        wr = d["w"] / total * 100 if total else 0
        avg_pnl = d["pnl"] / total if total else 0
        print(f"  {r:15s}: {total:4d} trades | WR: {wr:5.1f}% | Avg P&L: ${avg_pnl:8,.0f} | Total: ${d['pnl']:10,.0f}")

    # ---- PATTERN 3: By Gap Size ----
    print("\n" + "=" * 80)
    print("PATTERN 3: WIN RATE BY GAP SIZE")
    print("=" * 80)
    gap_bins = [
        ("10-20%", 10, 20),
        ("20-30%", 20, 30),
        ("30-50%", 30, 50),
        ("50-80%", 50, 80),
        ("80-120%", 80, 120),
        ("120%+", 120, 9999),
    ]
    for label, lo, hi in gap_bins:
        wt = [t for t in all_trades if lo <= t["gap_pct"] < hi]
        if not wt:
            continue
        w = sum(1 for t in wt if t["pnl"] > 0)
        wr = w / len(wt) * 100
        avg = sum(t["pnl"] for t in wt) / len(wt)
        total_pnl = sum(t["pnl"] for t in wt)
        print(f"  Gap {label:8s}: {len(wt):4d} trades | WR: {wr:5.1f}% | Avg P&L: ${avg:8,.0f} | Total: ${total_pnl:10,.0f}")

    # ---- PATTERN 4: By Entry Candle ----
    print("\n" + "=" * 80)
    print("PATTERN 4: WIN RATE BY ENTRY CANDLE (minutes after open)")
    print("=" * 80)
    candle_bins = [
        ("0-5 min", 0, 5),
        ("5-15 min", 5, 15),
        ("15-30 min", 15, 30),
        ("30-60 min", 30, 60),
        ("60-120 min", 60, 120),
        ("120+ min", 120, 9999),
    ]
    for label, lo, hi in candle_bins:
        wt = [t for t in all_trades if lo <= t["entry_candle"] < hi]
        if not wt:
            continue
        w = sum(1 for t in wt if t["pnl"] > 0)
        wr = w / len(wt) * 100
        avg = sum(t["pnl"] for t in wt) / len(wt)
        total_pnl = sum(t["pnl"] for t in wt)
        print(f"  {label:12s}: {len(wt):4d} trades | WR: {wr:5.1f}% | Avg P&L: ${avg:8,.0f} | Total: ${total_pnl:10,.0f}")

    # ---- PATTERN 5: By Hold Time ----
    print("\n" + "=" * 80)
    print("PATTERN 5: WIN RATE BY HOLD TIME")
    print("=" * 80)
    time_bins = [
        ("0-3 min", 0, 3),
        ("3-6 min", 3, 6),
        ("6-10 min", 6, 10),
        ("10-15 min", 10, 15),
        ("15-30 min", 15, 30),
        ("30-60 min", 30, 60),
        ("60+ min", 60, 9999),
    ]
    for label, lo, hi in time_bins:
        wt = [t for t in all_trades if t["hold_time_min"] is not None and lo <= t["hold_time_min"] < hi]
        if not wt:
            continue
        w = sum(1 for t in wt if t["pnl"] > 0)
        wr = w / len(wt) * 100
        avg = sum(t["pnl"] for t in wt) / len(wt)
        total_pnl = sum(t["pnl"] for t in wt)
        print(f"  {label:12s}: {len(wt):4d} trades | WR: {wr:5.1f}% | Avg P&L: ${avg:8,.0f} | Total: ${total_pnl:10,.0f}")

    # ---- PATTERN 6: By Float (L strategy) ----
    print("\n" + "=" * 80)
    print("PATTERN 6: WIN RATE BY FLOAT (Strategy L only)")
    print("=" * 80)
    l_trades = [t for t in all_trades if t["strategy"] == "L"]
    float_bins = [
        ("0-1M", 0, 1_000_000),
        ("1-5M", 1_000_000, 5_000_000),
        ("5-15M", 5_000_000, 15_000_000),
        ("15-25M", 15_000_000, 25_000_000),
    ]
    for label, lo, hi in float_bins:
        wt = [t for t in l_trades if lo <= t["float_shares"] < hi]
        if not wt:
            continue
        w = sum(1 for t in wt if t["pnl"] > 0)
        wr = w / len(wt) * 100
        avg = sum(t["pnl"] for t in wt) / len(wt)
        total_pnl = sum(t["pnl"] for t in wt)
        print(f"  Float {label:8s}: {len(wt):4d} trades | WR: {wr:5.1f}% | Avg P&L: ${avg:8,.0f} | Total: ${total_pnl:10,.0f}")

    # ---- PATTERN 7: Top 10 Winners vs Top 10 Losers ----
    print("\n" + "=" * 80)
    print("PATTERN 7: TOP 10 WINNERS")
    print("=" * 80)
    sorted_w = sorted(winners, key=lambda x: -x["pnl"])[:10]
    for t in sorted_w:
        print(f"  {t['date']} {t['ticker']:6s} {t['strategy']} gap={t['gap_pct']:5.1f}% candle={t['entry_candle']:3d} "
              f"hold={t['hold_time_min'] or 0:5.1f}m reason={t['reason']:12s} P&L=${t['pnl']:10,.0f}")

    print("\n" + "=" * 80)
    print("PATTERN 8: TOP 10 LOSERS")
    print("=" * 80)
    sorted_l = sorted(losers, key=lambda x: x["pnl"])[:10]
    for t in sorted_l:
        print(f"  {t['date']} {t['ticker']:6s} {t['strategy']} gap={t['gap_pct']:5.1f}% candle={t['entry_candle']:3d} "
              f"hold={t['hold_time_min'] or 0:5.1f}m reason={t['reason']:12s} P&L=${t['pnl']:10,.0f}")

    # ---- PATTERN 8: Time of Day ----
    print("\n" + "=" * 80)
    print("PATTERN 9: WIN RATE BY TIME OF DAY (entry candle proxy)")
    print("=" * 80)
    # Map candle numbers to approximate ET times
    # Candle 0 = 9:30, each candle = 2 min for 2-min bars
    time_of_day_bins = [
        ("9:30-9:45", 0, 7),
        ("9:45-10:00", 7, 15),
        ("10:00-10:30", 15, 30),
        ("10:30-11:30", 30, 60),
        ("11:30-13:00", 60, 90),
        ("13:00-14:00", 90, 120),
        ("14:00-15:00", 120, 150),
        ("15:00-16:00", 150, 180),
    ]
    for label, lo, hi in time_of_day_bins:
        wt = [t for t in all_trades if lo <= t["entry_candle"] < hi]
        if not wt:
            continue
        w = sum(1 for t in wt if t["pnl"] > 0)
        wr = w / len(wt) * 100
        avg = sum(t["pnl"] for t in wt) / len(wt)
        total_pnl = sum(t["pnl"] for t in wt)
        print(f"  {label:12s}: {len(wt):4d} trades | WR: {wr:5.1f}% | Avg P&L: ${avg:8,.0f} | Total: ${total_pnl:10,.0f}")

    # ---- KEY INSIGHTS ----
    print("\n" + "=" * 80)
    print("KEY INSIGHTS SUMMARY")
    print("=" * 80)

    # Find best/worst strategies
    best_strat = max(strat_data.items(), key=lambda x: x[1]["pnl"] / max(1, x[1]["w"] + x[1]["l"]))
    worst_strat = min(strat_data.items(), key=lambda x: x[1]["pnl"] / max(1, x[1]["w"] + x[1]["l"]))
    print(f"\n  Best strategy: {best_strat[0]} (avg P&L: ${best_strat[1]['pnl']/(best_strat[1]['w']+best_strat[1]['l']):,.0f}/trade)")
    print(f"  Worst strategy: {worst_strat[0]} (avg P&L: ${worst_strat[1]['pnl']/(worst_strat[1]['w']+worst_strat[1]['l']):,.0f}/trade)")

    # Find best/worst exit reasons
    best_reason = max(reason_data.items(), key=lambda x: x[1]["pnl"] / max(1, x[1]["w"] + x[1]["l"]))
    worst_reason = min(reason_data.items(), key=lambda x: x[1]["pnl"] / max(1, x[1]["w"] + x[1]["l"]))
    print(f"  Best exit reason: {best_reason[0]} (avg P&L: ${best_reason[1]['pnl']/(best_reason[1]['w']+best_reason[1]['l']):,.0f}/trade)")
    print(f"  Worst exit reason: {worst_reason[0]} (avg P&L: ${worst_reason[1]['pnl']/(worst_reason[1]['w']+worst_reason[1]['l']):,.0f}/trade)")

    # Winners avg hold time vs losers
    w_hold = [t["hold_time_min"] for t in winners if t["hold_time_min"] is not None]
    l_hold = [t["hold_time_min"] for t in losers if t["hold_time_min"] is not None]
    if w_hold and l_hold:
        print(f"  Avg hold time winners: {sum(w_hold)/len(w_hold):.1f} min")
        print(f"  Avg hold time losers: {sum(l_hold)/len(l_hold):.1f} min")

    # Winners avg gap vs losers
    w_gap = [t["gap_pct"] for t in winners if t["gap_pct"]]
    l_gap = [t["gap_pct"] for t in losers if t["gap_pct"]]
    if w_gap and l_gap:
        print(f"  Avg gap winners: {sum(w_gap)/len(w_gap):.1f}%")
        print(f"  Avg gap losers: {sum(l_gap)/len(l_gap):.1f}%")

    # Winners avg entry candle vs losers
    w_candle = [t["entry_candle"] for t in winners if t["entry_candle"]]
    l_candle = [t["entry_candle"] for t in losers if t["entry_candle"]]
    if w_candle and l_candle:
        print(f"  Avg entry candle winners: {sum(w_candle)/len(w_candle):.0f}")
        print(f"  Avg entry candle losers: {sum(l_candle)/len(l_candle):.0f}")

    # Restore defaults
    tgc.G_MIN_GAP_PCT = 30.0
    tgc.G_TARGET_PCT = 11.0
    tgc.G_TARGET2_PCT = 30.0
    tgc.G_PARTIAL_SELL_PCT = 0.0
    tgc.G_TIME_LIMIT_MINUTES = 10
    tgc.G_STOP_PCT = 0.0
    tgc.G_TRAIL_PCT = 0.0
    tgc.G_TRAIL_ACTIVATE_PCT = 0.0
    tgc.L_MIN_GAP_PCT = 30.0
    tgc.L_MAX_FLOAT = 15_000_000
    tgc.L_TIME_LIMIT_MINUTES = 70
    tgc.L_TRAIL_PCT = 1.0
    tgc.L_TRAIL_ACTIVATE_PCT = 2.0
    tgc.L_STOP_PCT = 14.0
    tgc.L_PARTIAL_SELL_PCT = 0.0


if __name__ == "__main__":
    run_analysis()
