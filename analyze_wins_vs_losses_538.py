"""
Analyze trial #538 wins vs losses on 2026 OOS — same setup as the forward test.
Uses test_full.load_all_picks, deployed config snapshot, same data dirs.
"""
if __name__ != "__main__":
    raise SystemExit

import json, sys
from pathlib import Path

sys.path.insert(0, ".")
import test_green_candle_combined as tgc
from test_full import load_all_picks, MARGIN_THRESHOLD
from optimize_combined import set_strategy_params, _build_param_snapshot, _param_lock
import pandas as pd
import numpy as np

OOS_DATA_DIRS = ["stored_data_mar_may_2026", "stored_data_jun_2026"]
OOS_DATE_LO = "2026-03-01"
OOS_DATE_HI = "2026-06-30"
STARTING_CASH = 25_000

# --- Simulator config (exact match to forward test / optuna_gl_trail) ---
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

# --- Merge base config (w21b) + 538 G/L overrides — same as Optuna objective ---
with open("config/trial_w21b_511_deploy.json") as f:
    base = json.load(f)
params = dict(base["params"])

with open("config/trial_gl_trail_538_deploy.json") as f:
    cfg = json.load(f)
params.update(cfg["params"])  # 538 G+L params override base

with _param_lock:
    set_strategy_params(params)
    snapshot = _build_param_snapshot()

# --- Load data ---
print("Loading OOS data...")
dirs = [d for d in OOS_DATA_DIRS if Path(d).exists()]
all_dates, picks_by_date = load_all_picks(dirs)
dates = sorted(d for d in all_dates if OOS_DATE_LO <= d <= OOS_DATE_HI)
print(f"  {len(dates)} days: {dates[0]} → {dates[-1]}")

# --- Run backtest, capture full trade detail ---
cash = float(STARTING_CASH)
trades = []

for d in dates:
    picks = picks_by_date.get(d, [])
    if not picks:
        continue
    cash_account = cash < MARGIN_THRESHOLD
    try:
        states, new_cash, unsettled, _ = tgc.simulate_day_combined(
            picks, cash, cash_account, params=snapshot
        )
    except Exception as e:
        print(f"  ERROR on {d}: {e}")
        continue

    for st in states:
        if not st.get("exit_reason") or not st.get("position_cost", 0):
            continue
        pnl = st.get("pnl", 0)
        cost = st["position_cost"]
        pct = pnl / cost * 100 if cost > 0 else 0

        # Get pick-level fields
        pick = next((p for p in picks if p.get("ticker") == st.get("ticker")), {})

        trades.append({
            "date":              d,
            "ticker":            st.get("ticker", ""),
            "strategy":          st.get("strategy", "?"),
            "exit_reason":       st["exit_reason"],
            "pnl":               pnl,
            "pct_return":        pct,
            "win":               pnl > 0,
            "position_cost":     cost,
            "gap_pct":           st.get("gap_pct", pick.get("gap_pct", 0)),
            "first_candle_body_pct": st.get("first_candle_body_pct", 0),
            "vol_capped":        st.get("vol_capped", False),
            "pm_volume":         pick.get("pm_volume", 0),
            "premarket_high":    pick.get("premarket_high", 0),
            "prev_close":        pick.get("prev_close", 0),
            "open_price":        pick.get("market_open", 0),
        })

    cash = new_cash + (unsettled if cash_account else 0)

df = pd.DataFrame(trades)

print(f"\n{'='*80}")
print(f"TRIAL #538 — 2026 OOS WIN/LOSS ANALYSIS ({OOS_DATE_LO} to {OOS_DATE_HI})")
print(f"{'='*80}")
print(f"Total trades: {len(df)}  Winners: {df['win'].sum()} ({df['win'].mean()*100:.1f}%)  Losers: {(~df['win']).sum()}")
print(f"Total P&L: ${df['pnl'].sum():,.0f}  Final equity: ${cash:,.0f}")

# --- By strategy ---
print(f"\n{'─'*60}")
print("BY STRATEGY")
print(f"{'─'*60}")
for s in sorted(df["strategy"].unique()):
    sub = df[df["strategy"] == s]
    w = sub[sub["win"]]; l = sub[~sub["win"]]
    print(f"  [{s}] {len(sub):>3} trades  WR={sub['win'].mean()*100:.1f}%  "
          f"P&L=${sub['pnl'].sum():>+10,.0f}  "
          f"avg_win=${w['pnl'].mean():,.0f}  avg_loss=${l['pnl'].mean():,.0f}" if len(l) else
          f"  [{s}] {len(sub):>3} trades  WR={sub['win'].mean()*100:.1f}%  P&L=${sub['pnl'].sum():>+10,.0f}")

# --- By exit reason ---
print(f"\n{'─'*60}")
print("BY EXIT REASON")
print(f"{'─'*60}")
for r in df["exit_reason"].value_counts().index:
    sub = df[df["exit_reason"] == r]
    print(f"  {r:<12} {len(sub):>3} trades  WR={sub['win'].mean()*100:.1f}%  avg=${sub['pnl'].mean():>+8,.0f}  total=${sub['pnl'].sum():>+10,.0f}")

# --- Gap % buckets ---
print(f"\n{'─'*60}")
print("GAP % BUCKETS")
print(f"{'─'*60}")
for lo, hi in [(10,20),(20,30),(30,50),(50,100),(100,500)]:
    b = df[(df["gap_pct"]>=lo)&(df["gap_pct"]<hi)]
    if len(b):
        print(f"  {lo:>3}-{hi:<3}%  {len(b):>3} trades  WR={b['win'].mean()*100:.1f}%  avg=${b['pnl'].mean():>+8,.0f}  total=${b['pnl'].sum():>+10,.0f}")

# --- First candle body (G only) ---
g = df[df["strategy"]=="G"]
if len(g):
    print(f"\n{'─'*60}")
    print("G ONLY — FIRST CANDLE BODY %")
    print(f"{'─'*60}")
    for lo, hi in [(0,2),(2,5),(5,10),(10,20),(20,100)]:
        b = g[(g["first_candle_body_pct"]>=lo)&(g["first_candle_body_pct"]<hi)]
        if len(b):
            print(f"  {lo:>2}-{hi:<3}%  {len(b):>3} trades  WR={b['win'].mean()*100:.1f}%  avg=${b['pnl'].mean():>+8,.0f}  total=${b['pnl'].sum():>+10,.0f}")

# --- Vol capped ---
print(f"\n{'─'*60}")
print("VOLUME CAPPED")
print(f"{'─'*60}")
for capped in [True, False]:
    b = df[df["vol_capped"]==capped]
    if len(b):
        print(f"  {'Vol-capped' if capped else 'Full size':<12}  {len(b):>3} trades  WR={b['win'].mean()*100:.1f}%  avg=${b['pnl'].mean():>+8,.0f}  total=${b['pnl'].sum():>+10,.0f}")

# --- PM volume quartiles ---
has_pm = df[df["pm_volume"]>0]
if len(has_pm):
    print(f"\n{'─'*60}")
    print("PM VOLUME QUARTILES (winners vs losers)")
    print(f"{'─'*60}")
    w_med = has_pm[has_pm["win"]]["pm_volume"].median()
    l_med = has_pm[~has_pm["win"]]["pm_volume"].median()
    print(f"  Winners median PM vol: {w_med/1e6:.1f}M")
    print(f"  Losers  median PM vol: {l_med/1e6:.1f}M")
    q = has_pm["pm_volume"].quantile([0.25,0.5,0.75])
    for label, lo, hi in [("Q1 (lowest)",0,q[0.25]),("Q2",q[0.25],q[0.5]),("Q3",q[0.5],q[0.75]),("Q4 (highest)",q[0.75],1e12)]:
        b = has_pm[(has_pm["pm_volume"]>=lo)&(has_pm["pm_volume"]<hi)]
        if len(b):
            print(f"  {label:<14}  {len(b):>3} trades  WR={b['win'].mean()*100:.1f}%  pm_vol_med={b['pm_volume'].median()/1e6:.1f}M  avg_pnl=${b['pnl'].mean():>+8,.0f}")

# --- Day of week ---
print(f"\n{'─'*60}")
print("DAY OF WEEK")
print(f"{'─'*60}")
df["dow"] = pd.to_datetime(df["date"]).dt.day_name()
for day in ["Monday","Tuesday","Wednesday","Thursday","Friday"]:
    b = df[df["dow"]==day]
    if len(b):
        print(f"  {day:<10}  {len(b):>3} trades  WR={b['win'].mean()*100:.1f}%  avg=${b['pnl'].mean():>+8,.0f}  total=${b['pnl'].sum():>+10,.0f}")

# --- Top losers ---
print(f"\n{'─'*60}")
print("TOP 15 BIGGEST LOSERS")
print(f"{'─'*60}")
for _, r in df[~df["win"]].nsmallest(15,"pnl").iterrows():
    print(f"  {r['date']} {r['ticker']:<6} [{r['strategy']}] gap={r['gap_pct']:.0f}% "
          f"{r['exit_reason']:<10} ${r['pnl']:>+9,.0f} ({r['pct_return']:+.1f}%)  "
          f"fc={r['first_candle_body_pct']:.1f}%  pm={r['pm_volume']/1e6:.1f}M  vc={'Y' if r['vol_capped'] else 'N'}")

# --- Top winners ---
print(f"\n{'─'*60}")
print("TOP 15 BIGGEST WINNERS")
print(f"{'─'*60}")
for _, r in df[df["win"]].nlargest(15,"pnl").iterrows():
    print(f"  {r['date']} {r['ticker']:<6} [{r['strategy']}] gap={r['gap_pct']:.0f}% "
          f"{r['exit_reason']:<10} ${r['pnl']:>+9,.0f} ({r['pct_return']:+.1f}%)  "
          f"fc={r['first_candle_body_pct']:.1f}%  pm={r['pm_volume']/1e6:.1f}M  vc={'Y' if r['vol_capped'] else 'N'}")

df.to_csv("trade_analysis_538_2026_oos.csv", index=False)
print(f"\nSaved to trade_analysis_538_2026_oos.csv")
