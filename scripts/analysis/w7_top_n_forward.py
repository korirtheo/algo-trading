"""Forward-test the top-N W7 trials on 2026 in PARALLEL and rank by forward PnL.

Usage:
  python scripts/analysis/w7_top_n_forward.py 10        # top 10 trials
  python scripts/analysis/w7_top_n_forward.py 20 --workers 6

Why: TPE often finds train-score peaks that overfit. The trial that
generalizes best is rarely the train-score #1; we observed #462 (top train)
forward 9× worse than #254 (#2 train). So pick by forward, not train.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import psycopg2


STARTING_CASH = 25_000
DATA_DIRS = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]
BASELINE = "config/trial_432_params.json"
OUTDIR = "results/walk_forward_v7_news"


def fetch_top_n(n):
    """Pull params for top N trials by training score."""
    c = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres", dbname="optuna")
    cur = c.cursor()
    cur.execute("""
        SELECT t.trial_id, t.number, tv.value
          FROM trials t JOIN trial_values tv ON tv.trial_id=t.trial_id
         WHERE t.state='COMPLETE' AND tv.value > 0
         ORDER BY tv.value DESC LIMIT %s
    """, (n,))
    top = cur.fetchall()

    trials = []
    for tid, num, score in top:
        cur.execute("SELECT key, value_json FROM trial_user_attributes WHERE trial_id=%s", (tid,))
        ua = dict(cur.fetchall())
        cur.execute(
            "SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=%s",
            (tid,))
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
        trials.append({
            "trial_id": tid, "number": num, "score": float(score),
            "user_attrs": ua, "params": params,
        })
    c.close()
    return trials


def forward_one(trial):
    """Run forward 2026 on a single trial — designed for ProcessPool."""
    # Lazy imports because they're heavy + this is the worker entrypoint
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

    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if d.startswith("2026")])

    cash = STARTING_CASH
    daily_eq = [cash]
    n_trades = 0
    worst_trade = 0
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
                if st["pnl"] < worst_trade: worst_trade = st["pnl"]
        cash = end_c + (unset if is_cash else 0)
        daily_eq.append(cash)
    eq = np.array(daily_eq)
    peak = np.maximum.accumulate(eq)
    dd_pct = float((eq - peak).min() / peak[(eq - peak).argmin()] * 100 if len(peak) else 0)
    dd_dollar = float((eq - peak).min())
    return {
        "trial_number": trial["number"],
        "train_score": trial["score"],
        "train_pnl": float(trial["user_attrs"].get("total_pnl", 0)),
        "train_pf": float(trial["user_attrs"].get("pf", 0)),
        "forward_final": float(eq[-1]),
        "forward_pnl": float(eq[-1] - STARTING_CASH),
        "forward_multi": float(eq[-1] / STARTING_CASH),
        "forward_max_dd_pct": dd_pct,
        "forward_max_dd_dollar": dd_dollar,
        "n_trades": n_trades,
        "worst_trade": float(worst_trade),
        "mar_ratio": float(abs(eq[-1] - STARTING_CASH) / max(abs(dd_dollar), 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=10)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    print(f"Pulling top {args.n} trials from Postgres...")
    trials = fetch_top_n(args.n)
    print(f"Got {len(trials)} trials. Top by train score:")
    for t in trials[:5]:
        print(f"  #{t['number']:<4} train_score=${t['score']:>12,.0f}")

    os.makedirs(OUTDIR, exist_ok=True)

    print(f"\nForward-testing in parallel with {args.workers} workers...")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(forward_one, t): t["number"] for t in trials}
        for fut in as_completed(futs):
            num = futs[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"  #{r['trial_number']:<4} train ${r['train_score']/1e6:>5.2f}M  "
                      f"-> forward ${r['forward_pnl']:>+9,.0f}  "
                      f"({r['forward_multi']:.2f}x)  DD {r['forward_max_dd_pct']:>5.1f}%  "
                      f"MAR {r['mar_ratio']:.2f}")
            except Exception as e:
                print(f"  #{num}: FAILED ({str(e)[:80]})")

    # Rank by forward PnL
    results.sort(key=lambda r: -r["forward_pnl"])

    print(f"\n{'='*90}")
    print(f"  Top {args.n} W7 trials — RANKED BY FORWARD 2026 PnL")
    print(f"{'='*90}")
    print(f"  {'rank':<5} {'trial':<6} {'train$':>10} {'fwd$':>11} {'fwd_x':>7} {'fwd_DD%':>9} {'MAR':>5} {'tr':>4}")
    print(f"  {'-'*5} {'-'*6} {'-'*10} {'-'*11} {'-'*7} {'-'*9} {'-'*5} {'-'*4}")
    for i, r in enumerate(results):
        print(f"  {i+1:>4}. #{r['trial_number']:<4} ${r['train_score']/1e6:>7.2f}M ${r['forward_pnl']:>+9,.0f} {r['forward_multi']:>6.2f}x {r['forward_max_dd_pct']:>8.1f}% {r['mar_ratio']:>5.2f} {r['n_trades']:>4}")

    out_path = f"{OUTDIR}/top_{args.n}_forward_ranked.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved {out_path}")

    if results:
        winner = results[0]
        print(f"\n  >>> BEST BY FORWARD: #{winner['trial_number']}  "
              f"+${winner['forward_pnl']:,.0f}  ({winner['forward_multi']:.2f}x)  "
              f"MAR {winner['mar_ratio']:.2f}")


if __name__ == "__main__":
    main()
