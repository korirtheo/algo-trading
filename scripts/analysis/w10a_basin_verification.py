"""Forward-test a diverse set of W10a trials to verify the basin is broad."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import psycopg2

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
DATA_DIRS_2026 = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
OUTDIR = "results/w10a_forward"

# Trials to test — mix of top by forward, mid by forward, and edge cases
TRIALS_TO_TEST = [
    (614, "top-by-fwd rank 2"),
    (619, "top-by-fwd rank 5"),
    (611, "top-by-fwd rank 8"),
    (724, "top by train (latest)"),
    (545, "mid-rank winner (train 130, fwd $828K)"),
    (578, "mid-rank winner (train 100, fwd $873K)"),
    (479, "mid-rank winner (train 132, fwd $888K)"),
    (428, "deep-rank winner (train 441, fwd $743K)"),
    (229, "deep-rank decent (train 352, fwd $301K)"),
    (172, "deep-rank decent (train 496, fwd $235K)"),
]


def fetch(trial_num):
    c = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres", dbname="optuna_w10")
    cur = c.cursor()
    cur.execute("""SELECT t.trial_id, tv.value FROM trials t
                   JOIN trial_values tv ON tv.trial_id=t.trial_id
                   WHERE t.state='COMPLETE' AND t.number=%s LIMIT 1""", (trial_num,))
    r = cur.fetchone()
    if not r: return None
    tid, score = r
    cur.execute("SELECT key, value_json FROM trial_user_attributes WHERE trial_id=%s", (tid,))
    ua = dict(cur.fetchall())
    cur.execute("SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=%s", (tid,))
    params = {}
    for name, val, dist in cur.fetchall():
        try:
            d = json.loads(dist) if dist else {}
            kind = d.get("name", "")
            if kind == "CategoricalDistribution":
                params[name] = d["attributes"]["choices"][int(val)]
            elif "Int" in kind:
                params[name] = int(val)
            else:
                params[name] = float(val)
        except Exception:
            params[name] = val
    c.close()
    return {"number": trial_num, "score": float(score), "user_attrs": ua, "params": params}


def forward(trial):
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD
    with open(BASELINE) as f: baseline = json.load(f)
    merged = dict(baseline); merged.update(trial["params"])
    for s in ALL_STRATS:
        merged[f"enable_{s}"] = (s in {"g", "l"})
    set_strategy_params(merged)
    tgc.USE_DYNAMIC_SLIPPAGE = True; tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True; tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0; tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08; tgc.MARGIN_MULTIPLIER = 1.0
    dirs = [d for d in DATA_DIRS_2026 if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if d.startswith("2026")])

    cash = STARTING_CASH; eq = [cash]; n = 0; g_n = 0; l_n = 0; g_pnl = 0; l_pnl = 0
    for d in dates:
        day_picks = picks_by_date.get(d, [])
        if not day_picks: continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(day_picks, cash, cash_account=is_cash)
        except Exception: continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                n += 1
                strat = st.get("strategy", "?")
                if strat == "G": g_n += 1; g_pnl += st["pnl"]
                elif strat == "L": l_n += 1; l_pnl += st["pnl"]
        cash = end_c + (unset if is_cash else 0)
        eq.append(cash)
    eq_arr = np.array(eq)
    peak = np.maximum.accumulate(eq_arr)
    dd = float((eq_arr - peak).min() / peak[(eq_arr - peak).argmin()] * 100) if peak.max() > 0 else 0
    return {
        "trial": trial["number"],
        "train_score": trial["score"],
        "g_min_gap_pct": trial["params"].get("g_min_gap_pct"),
        "l_min_gap": trial["params"].get("l_min_gap"),
        "forward_pnl": float(cash - STARTING_CASH),
        "forward_multi": float(cash / STARTING_CASH),
        "max_dd_pct": dd,
        "n_trades": n,
        "g_n": g_n, "g_pnl": float(g_pnl),
        "l_n": l_n, "l_pnl": float(l_pnl),
    }


def worker(arg):
    num, label = arg
    t = fetch(num)
    if t is None: return {"trial": num, "label": label, "error": "not found"}
    r = forward(t)
    r["label"] = label
    return r


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Forward-testing {len(TRIALS_TO_TEST)} W10a trials in parallel...\n")

    results = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(worker, t): t for t in TRIALS_TO_TEST}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                results.append(r)
                if "error" in r:
                    print(f"  #{r['trial']:<5} {r['label']:<45}  ERROR: {r['error']}")
                else:
                    print(f"  #{r['trial']:<5} {r['label']:<45}  g={r['g_min_gap_pct']}% l={r['l_min_gap']}%  "
                          f"fwd ${r['forward_pnl']:>+11,.0f}  ({r['forward_multi']:.2f}x)  DD {r['max_dd_pct']:.1f}%")
            except Exception as e:
                print(f"  failed: {e}")

    # Sort by forward
    results.sort(key=lambda r: -(r.get("forward_pnl", -1e10)))
    print(f"\n=== RANKED BY FORWARD ===")
    print(f"  {'#':<6} {'train$':>10} {'fwd$':>12} {'multi':>7} {'DD%':>7} {'g/l':<6} {'g_pnl':>11} {'l_pnl':>11}")
    for r in results:
        if "error" in r: continue
        gl = f"{r['g_n']}/{r['l_n']}"
        print(f"  #{r['trial']:<5} ${r['train_score']/1e6:>7.1f}M ${r['forward_pnl']:>+10,.0f} {r['forward_multi']:>6.2f}x "
              f"{r['max_dd_pct']:>6.1f}% {gl:<6} ${r['g_pnl']:>+9,.0f} ${r['l_pnl']:>+9,.0f}")

    # Stats
    fwds = [r["forward_pnl"] for r in results if "error" not in r]
    import statistics
    print(f"\n=== BASIN STATS ===")
    print(f"  Median fwd:   ${statistics.median(fwds):,.0f}")
    print(f"  Mean fwd:     ${statistics.mean(fwds):,.0f}")
    print(f"  Min / Max:    ${min(fwds):,.0f} / ${max(fwds):,.0f}")
    print(f"  Stdev:        ${statistics.stdev(fwds):,.0f}")
    print(f"  All positive: {sum(1 for f in fwds if f > 0)}/{len(fwds)}")

    with open(f"{OUTDIR}/basin_verification.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
