"""Check if L-wide #176 is a NEW strategy or just an L refinement.

For each on 2026 Jan-Jun:
  A) G #511 trades — what (ticker, date) G fires on
  B) L #511 trades — current L
  C) L #626 trades — prior Optuna L
  D) L #176 trades — new wide L

Then compute:
  - #176 unique trades (not in A, B, or C) — these are NEW alpha
  - #176 overlap with each
  - PnL breakdown of unique vs overlapping trades
If most #176 trades are NEW (not in A/B/C), it's a new strategy candidate.
If mostly overlapping with L variants, it's just an L tweak.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json, os
from collections import defaultdict
import psycopg2

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
DATA_DIRS = ["stored_data", "stored_data_oos", "stored_data_mar_may_2026",
             "stored_data_jun_2026", "stored_data_2026_gap_fill"]


def fetch_trial_params(db, study, num):
    c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname=db)
    cur = c.cursor()
    cur.execute('SELECT trial_id FROM trials WHERE number=%s AND study_id=(SELECT study_id FROM studies WHERE study_name=%s)', (num, study))
    tid = cur.fetchone()[0]
    cur.execute('SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=%s', (tid,))
    params = {}
    for name, val, dist in cur.fetchall():
        try:
            d = json.loads(dist) if dist else {}
            kind = d.get("name", "")
            if kind == "CategoricalDistribution": params[name] = d["attributes"]["choices"][int(val)]
            elif "Int" in kind: params[name] = int(val)
            else: params[name] = float(val)
        except: params[name] = float(val)
    c.close()
    return params


def collect_trades(params, only_strat):
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD
    set_strategy_params(params)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False
    tgc.MARGIN_MULTIPLIER = 1.0
    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    all_dates, picks = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if "2026-01-01" <= d <= "2026-06-30"])
    cash = STARTING_CASH
    trades = []
    for d in dates:
        dp = picks.get(d, [])
        if not dp: continue
        is_cash = cash < MARGIN_THRESHOLD
        try: states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except: continue
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0 and st.get("strategy") == only_strat:
                trades.append({"ticker": st.get("ticker"), "date": d,
                               "entry_time": st.get("entry_time"),
                               "exit_time": st.get("exit_time"),
                               "pnl": float(st.get("pnl") or 0)})
        cash = end_c + (unset if is_cash else 0)
    return trades


def main():
    with open(BASELINE) as f: bp = json.load(f)
    with open(W21B_DEPLOY) as f: base = json.load(f)

    print("=== Collect G #511 trades on 2026 ===")
    p = {**bp, **base["params"]}
    for s in ALL_STRATS: p[f"enable_{s}"] = (s == "g")
    g_trades = collect_trades(p, "G")
    g_pairs = set((t["ticker"], t["date"]) for t in g_trades)
    print(f"  {len(g_trades)} G trades, {len(g_pairs)} unique (ticker, date)")

    print("\n=== Collect L #511 trades ===")
    p = {**bp, **base["params"]}
    for s in ALL_STRATS: p[f"enable_{s}"] = (s == "l")
    l511_trades = collect_trades(p, "L")
    l511_pairs = set((t["ticker"], t["date"]) for t in l511_trades)
    print(f"  {len(l511_trades)} L #511 trades, {len(l511_pairs)} unique (ticker, date)")

    print("\n=== Collect L #626 trades ===")
    l626 = fetch_trial_params("optuna_l_only", "l_only_w21b", 626)
    p = {**bp, **base["params"]}
    for k, v in l626.items(): p[k] = v
    for s in ALL_STRATS: p[f"enable_{s}"] = (s == "l")
    l626_trades = collect_trades(p, "L")
    l626_pairs = set((t["ticker"], t["date"]) for t in l626_trades)
    print(f"  {len(l626_trades)} L #626 trades, {len(l626_pairs)} unique (ticker, date)")

    print("\n=== Collect L #176 trades (L-wide) ===")
    l176 = fetch_trial_params("optuna_l_wide", "l_wide_w21b", 176)
    p = {**bp, **base["params"]}
    for k, v in l176.items(): p[k] = v
    for s in ALL_STRATS: p[f"enable_{s}"] = (s == "l")
    l176_trades = collect_trades(p, "L")
    l176_pairs = set((t["ticker"], t["date"]) for t in l176_trades)
    print(f"  {len(l176_trades)} L #176 trades, {len(l176_pairs)} unique (ticker, date)")

    # Compute overlap of #176 with each strategy
    overlap_g    = l176_pairs & g_pairs
    overlap_l511 = l176_pairs & l511_pairs
    overlap_l626 = l176_pairs & l626_pairs
    new_only = l176_pairs - g_pairs - l511_pairs - l626_pairs

    print(f"\n{'='*90}")
    print(f"  #176 OVERLAP with existing strategies (by ticker, date)")
    print(f"{'='*90}")
    print(f"  Total #176 (ticker, date) pairs: {len(l176_pairs)}")
    print(f"    Overlap with G #511:           {len(overlap_g):>5} ({100*len(overlap_g)/len(l176_pairs):.1f}%)")
    print(f"    Overlap with L #511:           {len(overlap_l511):>5} ({100*len(overlap_l511)/len(l176_pairs):.1f}%)")
    print(f"    Overlap with L #626:           {len(overlap_l626):>5} ({100*len(overlap_l626)/len(l176_pairs):.1f}%)")
    print(f"    NEW (no overlap with any):     {len(new_only):>5} ({100*len(new_only)/len(l176_pairs):.1f}%)")

    # PnL breakdown
    pnl_total    = sum(t["pnl"] for t in l176_trades)
    pnl_new      = sum(t["pnl"] for t in l176_trades if (t["ticker"], t["date"]) in new_only)
    pnl_overlap_g    = sum(t["pnl"] for t in l176_trades if (t["ticker"], t["date"]) in overlap_g)
    pnl_overlap_l511 = sum(t["pnl"] for t in l176_trades if (t["ticker"], t["date"]) in overlap_l511)
    pnl_overlap_l626 = sum(t["pnl"] for t in l176_trades if (t["ticker"], t["date"]) in overlap_l626)

    print(f"\n  #176 PnL breakdown:")
    print(f"    Total PnL:          ${pnl_total:>+12,.0f}")
    print(f"    NEW-trade PnL:      ${pnl_new:>+12,.0f}  ({100*pnl_new/pnl_total if pnl_total else 0:.1f}%)")
    print(f"    Overlap-G PnL:      ${pnl_overlap_g:>+12,.0f}")
    print(f"    Overlap-L#511 PnL:  ${pnl_overlap_l511:>+12,.0f}")
    print(f"    Overlap-L#626 PnL:  ${pnl_overlap_l626:>+12,.0f}")

    # Same-bar conflict check on overlaps
    print(f"\n=== Time-overlap (capital conflict) on G overlapping (ticker, date) ===")
    g_by_pair = defaultdict(list)
    for t in g_trades: g_by_pair[(t["ticker"], t["date"])].append((t["entry_time"], t["exit_time"]))
    n176_inside_g = 0
    n176_after_g = 0
    for t in l176_trades:
        pair = (t["ticker"], t["date"])
        if pair not in overlap_g: continue
        l_ent = t["entry_time"]
        if not l_ent: continue
        for g_ent, g_ex in g_by_pair[pair]:
            if g_ent and g_ex and g_ent <= l_ent <= g_ex:
                n176_inside_g += 1
                break
        else:
            n176_after_g += 1
    print(f"  #176 trades that fire INSIDE G's holding window: {n176_inside_g}/{len(overlap_g)}")
    print(f"  #176 trades AFTER G's exit (no conflict):        {n176_after_g}/{len(overlap_g)}")

    # Verdict
    new_pct = 100*len(new_only)/len(l176_pairs) if l176_pairs else 0
    new_pnl_pct = 100*pnl_new/pnl_total if pnl_total else 0
    print(f"\n{'='*90}")
    print(f"  VERDICT")
    print(f"{'='*90}")
    if new_pct > 60:
        print(f"  #176 fires on {new_pct:.0f}% NEW (ticker, date) pairs not seen by G or any L variant")
        print(f"  This IS a candidate for a separate strategy slot.")
    elif new_pct > 30:
        print(f"  #176 has {new_pct:.0f}% NEW pairs — mixed; some additive value but mostly overlap.")
    else:
        print(f"  #176 mostly overlaps with existing strategies ({new_pct:.0f}% new). It's an L refinement, not a new strategy.")
    print(f"  ${pnl_new:,.0f} of #176's PnL comes from genuinely new trades ({new_pnl_pct:.0f}% of total).")


if __name__ == "__main__":
    main()
