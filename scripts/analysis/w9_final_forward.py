"""Forward-test the REAL W9 top 5 trials (post-overshoot) on 2026.

The orchestrator forward-tested at 600 trials and exited. Workers kept
running and produced 850+ more trials, finding much better basins.
This script pulls the current actual top 5 from Postgres optuna_w9
and forward-tests them in parallel.
"""
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
OUTDIR = "results/walk_forward_v9_cv"


def fetch_top_n(db, n=5):
    c = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres", dbname=db)
    cur = c.cursor()
    cur.execute("""
        SELECT t.trial_id, t.number, tv.value
          FROM trials t JOIN trial_values tv ON tv.trial_id=t.trial_id
         WHERE t.state='COMPLETE' AND tv.value > 0
         ORDER BY tv.value DESC LIMIT %s
    """, (n,))
    rows = cur.fetchall()
    out = []
    for tid, num, score in rows:
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
        out.append({"trial_id": tid, "number": num, "score": float(score),
                    "user_attrs": ua, "params": params})
    c.close()
    return out


def forward(trial):
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD

    with open(BASELINE) as f: baseline = json.load(f)
    merged = dict(baseline); merged.update(trial["params"])
    set_strategy_params(merged)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.MARGIN_MULTIPLIER = 1.0

    dirs = [d for d in DATA_DIRS_2026 if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if d.startswith("2026")])

    cash = STARTING_CASH
    daily_eq = [cash]
    n_trades = 0
    for d in dates:
        day_picks = picks_by_date.get(d, [])
        if not day_picks:
            daily_eq.append(cash); continue
        if tgc.NEWS_FILTER_ENABLED and (tgc.NEWS_MIN_ARTICLES > 0 or tgc.NEWS_REQUIRE_CATALYST):
            from news_filter import filter_picks as nfp
            day_picks = nfp(day_picks, d, tgc.NEWS_MIN_ARTICLES, tgc.NEWS_REQUIRE_CATALYST)
        if not day_picks:
            daily_eq.append(cash); continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception:
            daily_eq.append(cash); continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                n_trades += 1
        cash = end_c + (unset if is_cash else 0)
        daily_eq.append(cash)
    eq = np.array(daily_eq)
    peak = np.maximum.accumulate(eq)
    dd_pct = float((eq - peak).min() / peak[(eq - peak).argmin()] * 100) if len(peak) and peak[(eq - peak).argmin()] > 0 else 0
    return {
        "trial_number": trial["number"],
        "score": float(trial["score"]),
        "user_attrs": trial["user_attrs"],
        "forward_final": float(eq[-1]),
        "forward_pnl": float(eq[-1] - STARTING_CASH),
        "forward_multi": float(eq[-1] / STARTING_CASH),
        "forward_max_dd_pct": dd_pct,
        "forward_n_trades": n_trades,
    }


def main():
    print("Fetching top 5 from optuna_w9...")
    trials = fetch_top_n("optuna_w9", n=5)
    print(f"Got {len(trials)} trials:")
    for t in trials:
        ua = t["user_attrs"]
        print(f"  #{t['number']:<5} score=${t['score']:>10,.0f}  sum_pnl={ua.get('sum_pnl','-'):>10}  "
              f"min_pf={ua.get('min_pf','-'):>6}  n={ua.get('n','-'):>5}")

    os.makedirs(OUTDIR, exist_ok=True)

    print(f"\nForward-testing top 5 on 2026 (4 parallel workers)...")
    forwards = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(forward, t): t["number"] for t in trials}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                forwards.append(r)
                print(f"  #{r['trial_number']:<5} fwd ${r['forward_pnl']:>+9,.0f}  "
                      f"({r['forward_multi']:.2f}x)  DD {r['forward_max_dd_pct']:.1f}%  "
                      f"trades={r['forward_n_trades']}")
            except Exception as e:
                print(f"  forward failed: {e}")
    forwards.sort(key=lambda r: -r["forward_pnl"])

    out_path = f"{OUTDIR}/w9_final_top5_forward.json"
    with open(out_path, "w") as f:
        json.dump(forwards, f, indent=2)
    print(f"\nWrote {out_path}")

    if forwards:
        w = forwards[0]
        print(f"\n=== W9 WINNER BY FORWARD: #{w['trial_number']} ===")
        print(f"  Forward 2026 PnL: ${w['forward_pnl']:+,.0f} ({w['forward_multi']:.2f}x)")
        print(f"  vs W7 #254 (+$199,553): ${w['forward_pnl'] - 199553:+,.0f}")
        print(f"  vs W8 #1550 (+$36,668): ${w['forward_pnl'] - 36668:+,.0f}")
        print(f"  vs #124 deployed (+$22,534): ${w['forward_pnl'] - 22534:+,.0f}")


if __name__ == "__main__":
    main()
