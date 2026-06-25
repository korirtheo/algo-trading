"""Multi-dimensional pattern analysis of #124's losers across 2024-26.

For every closed trade, captures:
  - Stock context:    price tier, float, gap %, premarket $vol
  - Volatility:       1st-candle body %, pre-entry ATR, intraday range
  - Participation:    % of V_eff_adj, % of cumulative $vol, vol-capped flag
  - Slippage:         modeled entry slip bp, modeled exit slip bp
  - Trade ID:         strategy, entry time-of-day, exit reason, holding time
  - Day context:      shape classification, num picks, regime

Then runs heterogeneity analysis: where does the loss distribution concentrate?
Adaptive controls should target the dimensions where heterogeneity is real.

Outputs:
  - Console report with per-dimension breakdowns and flags
  - Full trade-level CSV at results/wf_pf_microcap_pump_noX/124_loss_patterns.csv
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import csv
import numpy as np
import pandas as pd
from collections import defaultdict

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD
from scripts.analysis.squeeze_taxonomy_2021_2026 import day_signature, classify_shape
from strategies.regime_gate import classify_regime, compute_features as _regime_feats

CONFIG = "config/trial_124_microcap_pump_extracted.json"
BASELINE = "config/trial_432_params.json"
STARTING_CASH = 25_000
OUT_DIR = "results/wf_pf_microcap_pump_noX"
CSV_PATH = os.path.join(OUT_DIR, "124_loss_patterns.csv")

YEAR_DIRS = {
    "2024": ["stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
              "stored_data_jul_sep_2024", "stored_data_oct_dec_2024"],
    "2025": ["stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
              "stored_data_jul_2025", "stored_data_oos"],
    "2026": ["stored_data", "stored_data_mar_may_2026"],
}


def _merged(p):
    with open(BASELINE) as f: b = json.load(f)
    m = dict(b); m.update(p); return m


def _classify_day(picks):
    try:
        feats = _regime_feats(picks)
        regime = classify_regime(feats)
    except Exception:
        regime = "NORMAL"
    sig = day_signature(picks)
    if sig is None: return "empty"
    return classify_shape(sig, regime)


def _price_tier(p):
    if p < 1: return "<$1"
    if p < 3: return "$1-3"
    if p < 10: return "$3-10"
    if p < 30: return "$10-30"
    return "$30+"


def _atr_pct_pre_entry(mh, entry_ts, n_bars=10):
    """ATR as % of close on the n bars BEFORE entry. None if too few bars."""
    if mh is None or len(mh) == 0 or entry_ts is None:
        return None
    pre = mh.loc[mh.index < entry_ts]
    if len(pre) < 3:
        return None
    pre = pre.tail(n_bars)
    if len(pre) < 2:
        return None
    # True range = max(high - low, |high - prev_close|, |low - prev_close|)
    prev_close = pre["Close"].shift(1)
    tr1 = pre["High"] - pre["Low"]
    tr2 = (pre["High"] - prev_close).abs()
    tr3 = (pre["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.mean()
    last_close = float(pre["Close"].iloc[-1])
    return float(atr / last_close * 100) if last_close > 0 else None


def _time_of_day_bucket(ts):
    if ts is None: return "unk"
    try:
        h, m = ts.hour, ts.minute
        mins = h * 60 + m
    except Exception:
        return "unk"
    # 9:30-10:00 = "open", 10:00-12:00 = "morning", 12:00-14:00 = "midday",
    # 14:00-15:30 = "afternoon", 15:30-16:00 = "close"
    if mins < 600: return "open"
    if mins < 720: return "morning"
    if mins < 840: return "midday"
    if mins < 930: return "afternoon"
    return "close"


def _bucket(value, edges, labels):
    if value is None: return "unk"
    for i, e in enumerate(edges):
        if value < e: return labels[i]
    return labels[-1]


def collect_trades(year, dirs):
    dirs_present = [d for d in dirs if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs_present)
    test_dates = [d for d in all_dates if d.startswith(year)]
    cash = STARTING_CASH
    trades = []
    for d in test_dates:
        day_picks = picks_by_date.get(d, [])
        if not day_picks: continue
        shape = _classify_day(day_picks)
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception:
            continue
        for st in states:
            if st.get("exit_reason") is None or st.get("position_cost", 0) <= 0:
                continue
            mh = st["mh"]
            entry_ts = st.get("entry_time")
            # entry_price on state is NULLED after exit. Look it up from the
            # market-hour candles using entry_time instead.
            entry_price = st.get("entry_price")
            if (entry_price is None or entry_price == 0) and entry_ts is not None and mh is not None:
                try:
                    if entry_ts in mh.index:
                        entry_price = float(mh.loc[entry_ts]["Open"])
                    else:
                        # Find nearest bar at or before entry_ts
                        candidates = mh.loc[mh.index <= entry_ts]
                        if len(candidates) > 0:
                            entry_price = float(candidates.iloc[-1]["Open"])
                except Exception:
                    pass
            exit_price = st.get("exit_price")
            position_cost = st["position_cost"]
            shares = st.get("shares") or 0
            pnl = st["pnl"]
            pnl_pct = (pnl / position_cost * 100) if position_cost > 0 else 0

            # Pre-entry features
            atr_pct = _atr_pct_pre_entry(mh, entry_ts)
            # First-candle body % (already on state)
            first_body = st.get("first_candle_body_pct", 0.0)

            # Volume: cumulative $-volume from open to entry
            cum_dvol = None
            if entry_ts is not None and mh is not None:
                pre = mh.loc[mh.index <= entry_ts]
                if len(pre) > 0:
                    cum_dvol = float((pre["Close"] * pre["Volume"]).sum())

            # Participation: pos / V_eff_adj
            participation_pct = None
            if entry_ts is not None and mh is not None:
                try:
                    v_eff_adj, _, _, _ = tgc._multi_window_effective_volume(
                        mh, entry_ts, entry_price or 0)
                    if v_eff_adj > 0:
                        participation_pct = position_cost / v_eff_adj * 100
                except Exception:
                    pass

            # Modeled entry slippage
            slip_in_bp = None
            try:
                if entry_price and position_cost > 0 and mh is not None and entry_ts is not None:
                    v_eff_adj, _, _, _ = tgc._multi_window_effective_volume(
                        mh, entry_ts, entry_price)
                    _slip_pct = tgc._entry_slip_pct(entry_price, position_cost, v_eff_adj)
                    slip_in_bp = _slip_pct * 100
            except Exception:
                pass

            # Holding time
            exit_ts = st.get("exit_time")
            holding_min = None
            if entry_ts is not None and exit_ts is not None:
                try:
                    holding_min = (exit_ts - entry_ts).total_seconds() / 60
                except Exception:
                    pass

            tier = _price_tier(entry_price or 0)
            tod = _time_of_day_bucket(entry_ts)

            trades.append({
                "year": year, "date": d, "ticker": st["ticker"],
                "strategy": st.get("strategy"),
                "shape": shape,
                "entry_price": entry_price, "exit_price": exit_price,
                "position_cost": position_cost, "shares": shares,
                "pnl": pnl, "pnl_pct": pnl_pct,
                "exit_reason": st.get("exit_reason"),
                "holding_min": holding_min,
                "price_tier": tier,
                "tod_bucket": tod,
                "gap_pct": st.get("gap_pct", 0),
                "pm_volume": st.get("pm_volume", 0),
                "first_body_pct": first_body,
                "atr_pct_pre": atr_pct,
                "participation_pct": participation_pct,
                "slip_in_bp": slip_in_bp,
                "vol_capped": st.get("vol_capped", False),
                "cum_dollar_vol": cum_dvol,
                "pm_dollar_vol": (st.get("pm_volume", 0) * (entry_price or 0)),
            })
        cash = end_c
        if is_cash: cash += unset
    return trades, cash


def report_dimension(df, dim, label, min_n=20):
    """Per-bucket win rate, avg pnl %, avg loss size."""
    print(f"\n{label} ({dim}):")
    print(f"  {'bucket':<14} {'n':>5} {'wr%':>6} {'avg_pnl%':>9} {'avg_win%':>9} {'avg_loss%':>10} {'med_loss%':>10}")
    g = df.groupby(dim)
    rows = []
    for bucket, sub in g:
        n = len(sub)
        if n < min_n: continue
        wr = (sub["pnl"] > 0).mean() * 100
        avg = sub["pnl_pct"].mean()
        wins = sub[sub["pnl"] > 0]["pnl_pct"]
        losses = sub[sub["pnl"] < 0]["pnl_pct"]
        avg_win = wins.mean() if len(wins) else 0
        avg_loss = losses.mean() if len(losses) else 0
        med_loss = losses.median() if len(losses) else 0
        rows.append((bucket, n, wr, avg, avg_win, avg_loss, med_loss))
    rows.sort(key=lambda r: -r[1])  # by n
    for bucket, n, wr, avg, aw, al, ml in rows:
        print(f"  {str(bucket):<14} {n:>5} {wr:>5.1f}% {avg:>+8.2f}% "
              f"{aw:>+8.2f}% {al:>+9.2f}% {ml:>+9.2f}%")
    return rows


def heterogeneity_flag(rows):
    """Are the buckets meaningfully different in avg_pnl_pct?"""
    if len(rows) < 2: return None
    pcts = [r[3] for r in rows]  # avg_pnl%
    spread = max(pcts) - min(pcts)
    if spread >= 5:
        return f"STRONG (spread {spread:.1f}pp avg-pnl across buckets)"
    if spread >= 2:
        return f"moderate (spread {spread:.1f}pp)"
    return f"weak (spread {spread:.1f}pp — bucket choice doesn't matter much)"


def cross_tab(df, dim1, dim2, label, min_n=15):
    """Cross-tab of avg pnl_pct across two dimensions."""
    print(f"\n{label} (avg pnl%):")
    pivot = df.groupby([dim1, dim2]).agg(
        n=("pnl", "count"),
        avg_pnl=("pnl_pct", "mean"),
        wr=("pnl", lambda s: (s > 0).mean() * 100),
    ).reset_index()
    pivot = pivot[pivot["n"] >= min_n]
    if len(pivot) == 0:
        print("  (no cells with >= 20 trades)")
        return
    print(f"  {dim1:<10} {dim2:<14} {'n':>5} {'wr%':>6} {'avg_pnl%':>9}")
    pivot = pivot.sort_values("avg_pnl")
    for _, r in pivot.iterrows():
        print(f"  {str(r[dim1]):<10} {str(r[dim2]):<14} {int(r['n']):>5} "
              f"{r['wr']:>5.1f}% {r['avg_pnl']:>+8.2f}%")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(CONFIG) as f: cfg = json.load(f)
    set_strategy_params(_merged(cfg["params"]))
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.X_MIN_FIRST_LEG_GAIN_PCT = 9999.0

    all_trades = []
    for year, dirs in YEAR_DIRS.items():
        print(f"\nCollecting {year}...")
        trades, _ = collect_trades(year, dirs)
        print(f"  {len(trades)} trades")
        all_trades.extend(trades)

    df = pd.DataFrame(all_trades)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nWrote {len(df)} trades to {CSV_PATH}")

    # Bucket the continuous features
    df["atr_bucket"] = df["atr_pct_pre"].apply(
        lambda v: _bucket(v, [2, 5, 10, 20], ["<2%", "2-5%", "5-10%", "10-20%", ">20%"]))
    df["participation_bucket"] = df["participation_pct"].apply(
        lambda v: _bucket(v, [1, 3, 8, 15], ["<1%", "1-3%", "3-8%", "8-15%", ">=15%"]))
    df["slip_bucket"] = df["slip_in_bp"].apply(
        lambda v: _bucket(v, [10, 30, 60, 100], ["<10bp", "10-30bp", "30-60bp", "60-100bp", ">100bp"]))
    df["gap_bucket"] = df["gap_pct"].apply(
        lambda v: _bucket(v, [15, 30, 60, 100], ["<15%", "15-30%", "30-60%", "60-100%", ">100%"]))
    df["cum_dvol_bucket"] = df["cum_dollar_vol"].apply(
        lambda v: _bucket(v, [100_000, 500_000, 2_000_000, 10_000_000],
                          ["<100K", "100K-500K", "500K-2M", "2M-10M", ">10M"]))
    df["pm_dvol_bucket"] = df["pm_dollar_vol"].apply(
        lambda v: _bucket(v, [500_000, 2_000_000, 10_000_000, 50_000_000],
                          ["<500K", "500K-2M", "2M-10M", "10M-50M", ">50M"]))

    print(f"\n{'='*80}")
    print(f"#124 TRADE PATTERN ANALYSIS — {len(df)} trades, 2024-26")
    print(f"  Overall: WR {(df['pnl']>0).mean()*100:.1f}%  avg_pnl% {df['pnl_pct'].mean():+.2f}%  "
          f"total $ {df['pnl'].sum():+,.0f}")
    print(f"{'='*80}")

    rows_strat = report_dimension(df, "strategy", "BY STRATEGY")
    rows_tier  = report_dimension(df, "price_tier", "BY PRICE TIER")
    rows_atr   = report_dimension(df, "atr_bucket", "BY PRE-ENTRY ATR%")
    rows_part  = report_dimension(df, "participation_bucket", "BY PARTICIPATION (% of V_eff_adj)")
    rows_slip  = report_dimension(df, "slip_bucket", "BY MODELED ENTRY SLIPPAGE")
    rows_tod   = report_dimension(df, "tod_bucket", "BY TIME-OF-DAY ENTRY")
    rows_gap   = report_dimension(df, "gap_bucket", "BY GAP %")
    rows_shape = report_dimension(df, "shape", "BY DAY SHAPE")
    rows_cvol  = report_dimension(df, "cum_dvol_bucket", "BY CUMULATIVE $-VOLUME at entry")
    rows_pmvol = report_dimension(df, "pm_dvol_bucket", "BY PREMARKET $-VOLUME")

    print(f"\n{'='*80}")
    print("HETEROGENEITY VERDICTS (does adapting on this dim matter?)")
    print(f"{'='*80}")
    for label, rows in [
        ("strategy", rows_strat), ("price_tier", rows_tier),
        ("atr", rows_atr), ("participation", rows_part),
        ("slip", rows_slip), ("tod", rows_tod), ("gap", rows_gap), ("shape", rows_shape),
        ("cum_$vol", rows_cvol), ("pm_$vol", rows_pmvol),
    ]:
        v = heterogeneity_flag(rows) if rows else "(insufficient data)"
        print(f"  {label:<14} -> {v}")

    # Cross-tabs of the strongest signals
    print(f"\n{'='*80}")
    print("CROSS-TABS — where do losses concentrate?")
    print(f"{'='*80}")
    cross_tab(df, "strategy", "price_tier", "Strategy x Price Tier")
    cross_tab(df, "strategy", "atr_bucket", "Strategy x ATR%")
    cross_tab(df, "strategy", "participation_bucket", "Strategy x Participation")
    cross_tab(df, "strategy", "tod_bucket", "Strategy x Time-of-Day")
    cross_tab(df, "shape", "strategy", "Shape x Strategy")

    # --- WIN-vs-LOSS FEATURE CONTRAST ---
    # For each dimension, what's the WIN trade distribution vs LOSS trade
    # distribution? Different distributions => the dim discriminates wins.
    print(f"\n{'='*80}")
    print("WIN vs LOSS — what features make a winner different from a loser?")
    print(f"{'='*80}")
    winners = df[df["pnl"] > 0]
    losers  = df[df["pnl"] < 0]
    print(f"  {len(winners)} winners ({100*len(winners)/len(df):.1f}%), "
          f"{len(losers)} losers ({100*len(losers)/len(df):.1f}%)\n")

    def _feature_contrast(col, name):
        w = winners[col].dropna()
        l = losers[col].dropna()
        if len(w) < 20 or len(l) < 20:
            print(f"  {name:<22} insufficient data"); return
        wm, lm = w.mean(), l.mean()
        delta = wm - lm
        print(f"  {name:<22} winners avg={wm:>10.2f}  losers avg={lm:>10.2f}  "
              f"delta={delta:+10.2f}  ratio={wm/lm if lm else float('inf'):.2f}x")

    _feature_contrast("entry_price", "entry_price ($)")
    _feature_contrast("gap_pct", "gap_pct")
    _feature_contrast("pm_volume", "pm_volume")
    _feature_contrast("pm_dollar_vol", "pm_dollar_vol")
    _feature_contrast("cum_dollar_vol", "cum_$vol at entry")
    _feature_contrast("atr_pct_pre", "pre-entry ATR%")
    _feature_contrast("participation_pct", "participation%")
    _feature_contrast("slip_in_bp", "modeled slip (bp)")
    _feature_contrast("first_body_pct", "first-candle body%")
    _feature_contrast("holding_min", "holding minutes")
    _feature_contrast("position_cost", "position_cost ($)")

    # Per-bucket WR side-by-side
    print(f"\n  WIN-RATE PER BUCKET (>= 30 trades):")
    for dim, label in [
        ("strategy", "strategy"), ("price_tier", "price_tier"),
        ("atr_bucket", "atr"), ("participation_bucket", "participation"),
        ("cum_dvol_bucket", "cum_$vol"), ("pm_dvol_bucket", "pm_$vol"),
        ("tod_bucket", "tod"), ("shape", "shape"),
    ]:
        g = df.groupby(dim).agg(
            n=("pnl", "count"),
            wr=("pnl", lambda s: (s > 0).mean() * 100),
            avg=("pnl_pct", "mean"),
        ).reset_index()
        g = g[g["n"] >= 30].sort_values("wr", ascending=False)
        if len(g) == 0: continue
        print(f"\n    {label.upper()}")
        for _, r in g.iterrows():
            print(f"      {str(r[dim]):<14} n={int(r['n']):>4}  "
                  f"WR={r['wr']:>5.1f}%  avg={r['avg']:>+6.2f}%")

    # Exit-reason analysis on losers
    print(f"\n{'='*80}")
    print("EXIT REASONS — what mechanism is exiting at a loss?")
    print(f"{'='*80}")
    losers = df[df["pnl"] < 0]
    print(f"  {len(losers)} losers / {len(df)} total ({100*len(losers)/len(df):.1f}%)")
    er = losers.groupby("exit_reason").agg(
        n=("pnl", "count"),
        avg=("pnl_pct", "mean"),
        worst=("pnl_pct", "min"),
        total=("pnl", "sum"),
    ).sort_values("total")
    for er_name, row in er.iterrows():
        print(f"  {str(er_name):<28} n={int(row['n']):>4}  avg={row['avg']:>+7.2f}%  "
              f"worst={row['worst']:>+7.2f}%  total=${row['total']:>+10,.0f}")


if __name__ == "__main__":
    main()
