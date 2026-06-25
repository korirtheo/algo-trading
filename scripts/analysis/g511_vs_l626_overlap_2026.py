"""Compute G #511 vs L #626 overlap on 2026.

Three layers:
  1) (ticker, date) pair overlap — how often both want the same name same day
  2) Time-window overlap — does L fire INSIDE G's [entry_ts, exit_ts]?
  3) PnL impact — what would the overlapping trades contribute
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


def fetch_l626():
    c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname='optuna_l_only')
    cur = c.cursor()
    cur.execute('SELECT trial_id FROM trials WHERE number=%s AND study_id=(SELECT study_id FROM studies WHERE study_name=%s)', (626, 'l_only_w21b'))
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


def setup_tgc(params):
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
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
    return tgc


def collect_trades(params, only_strat):
    from test_full import load_all_picks, MARGIN_THRESHOLD
    tgc = setup_tgc(params)
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
    print("=== 1. Get G #511 trades on 2026 ===")
    with open(BASELINE) as f: bp = json.load(f)
    with open(W21B_DEPLOY) as f: base = json.load(f)
    p511 = {**bp, **base["params"]}
    for s in ALL_STRATS: p511[f"enable_{s}"] = (s == "g")  # G only
    g_trades = collect_trades(p511, "G")
    print(f"  {len(g_trades)} G trades on 2026")

    # Group G trades by (ticker, date) -> list of (entry, exit) windows
    g_by_pair = defaultdict(list)
    for t in g_trades:
        g_by_pair[(t["ticker"], t["date"])].append((t["entry_time"], t["exit_time"], t["pnl"]))
    print(f"  {len(g_by_pair)} unique (ticker, date) pairs")

    print("\n=== 2. Get L #626 trades on 2026 ===")
    l626 = fetch_l626()
    p_l = dict(bp)
    for k, v in l626.items(): p_l[k] = v
    for s in ALL_STRATS: p_l[f"enable_{s}"] = (s == "l")  # L only
    l_trades = collect_trades(p_l, "L")
    print(f"  {len(l_trades)} L trades on 2026")

    l_by_pair = defaultdict(list)
    for t in l_trades:
        l_by_pair[(t["ticker"], t["date"])].append((t["entry_time"], t["exit_time"], t["pnl"]))
    print(f"  {len(l_by_pair)} unique (ticker, date) pairs")

    print("\n=== 3. Overlap analysis ===")
    overlap_pairs = set(g_by_pair.keys()) & set(l_by_pair.keys())
    print(f"  Same (ticker, date) pairs where BOTH G and L fired: {len(overlap_pairs)}")
    print(f"    As % of G trade-days: {len(overlap_pairs)/len(g_by_pair)*100:.1f}%")
    print(f"    As % of L trade-days: {len(overlap_pairs)/len(l_by_pair)*100:.1f}%")

    # Time-window overlap
    n_inside_g = 0; n_after_g = 0; n_before_g = 0
    l_pnl_inside = 0; l_pnl_after = 0; l_pnl_before = 0
    for tkr_date in overlap_pairs:
        g_windows = g_by_pair[tkr_date]
        l_entries = l_by_pair[tkr_date]
        for l_ent, l_ex, l_pnl in l_entries:
            inside = False
            for g_ent, g_ex, _ in g_windows:
                if g_ent and g_ex and l_ent and g_ent <= l_ent <= g_ex:
                    inside = True; break
            if inside:
                n_inside_g += 1; l_pnl_inside += l_pnl
            elif l_ent and g_windows and all(g_ex and l_ent > g_ex for _, g_ex, _ in g_windows):
                n_after_g += 1; l_pnl_after += l_pnl
            else:
                n_before_g += 1; l_pnl_before += l_pnl

    total_overlap_l = n_inside_g + n_after_g + n_before_g
    print(f"\n  L entries on G-overlap days: {total_overlap_l}")
    if total_overlap_l:
        print(f"    L INSIDE G window (capital conflict): {n_inside_g}  PnL ${l_pnl_inside:>+10,.0f}")
        print(f"    L AFTER G exit (no conflict):         {n_after_g}  PnL ${l_pnl_after:>+10,.0f}")
        print(f"    L BEFORE G entry (no conflict):       {n_before_g}  PnL ${l_pnl_before:>+10,.0f}")

    # Same-bar conflict count (within ~2 minutes)
    n_same_bar = 0
    for tkr_date in overlap_pairs:
        for g_ent, _, _ in g_by_pair[tkr_date]:
            for l_ent, _, _ in l_by_pair[tkr_date]:
                if g_ent and l_ent and abs((l_ent - g_ent).total_seconds()) < 120:
                    n_same_bar += 1; break

    print(f"\n  Same-bar (within 2 min) G+L entries on same ticker: {n_same_bar}")

    print(f"\n=== 4. Summary ===")
    print(f"  G fires on {len(g_by_pair)} unique (ticker, date) pairs")
    print(f"  L #626 fires on {len(l_by_pair)} unique (ticker, date) pairs")
    print(f"  Overlap: {len(overlap_pairs)} pairs ({len(overlap_pairs)/len(l_by_pair)*100:.1f}% of L days)")
    print(f"  Of those overlaps, {n_inside_g} L entries would CONFLICT (fire while G holds)")
    print(f"  In live engine with single position-per-ticker: those {n_inside_g} L trades blocked")
    print(f"  Total L PnL on conflict trades: ${l_pnl_inside:>+,.0f}")


if __name__ == "__main__":
    main()
