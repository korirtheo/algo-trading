"""Pull the current best W7 trial from Postgres and forward-test it on 2026.

Run while the W7 study is still going to get an early read on whether the
training peaks generalize.

Output:
  config/trial_<N>_w7_extracted.json   (full params, for later re-runs)
  results/walk_forward_v7_news/W5_train_2022..2025_test_2026_forward_PARTIAL.json
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os

import psycopg2
import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD


STORAGE = "postgresql://postgres@127.0.0.1:5432/optuna"
BASELINE = "config/trial_432_params.json"
DATA_DIRS_2026 = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]
STARTING_CASH = 25_000
OUTDIR = "results/walk_forward_v7_news"


def fetch_best_params():
    c = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres", dbname="optuna")
    cur = c.cursor()
    cur.execute("""
        SELECT t.trial_id, t.number, tv.value
          FROM trials t JOIN trial_values tv ON tv.trial_id = t.trial_id
         WHERE t.state = 'COMPLETE'
         ORDER BY tv.value DESC LIMIT 1
    """)
    tid, num, score = cur.fetchone()
    cur.execute(
        "SELECT key, value_json FROM trial_user_attributes WHERE trial_id=%s",
        (tid,),
    )
    ua = {k: v for k, v in cur.fetchall()}
    cur.execute(
        "SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=%s",
        (tid,),
    )
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
    return num, score, ua, params


def forward_2026(params, label):
    with open(BASELINE) as f:
        baseline = json.load(f)
    merged = dict(baseline)
    merged.update(params)
    set_strategy_params(merged)

    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    # (no hardcoded X kill — set_strategy_params already honored params["enable_x"])

    dirs = [d for d in DATA_DIRS_2026 if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if d.startswith("2026")])
    print(f"  forward 2026: {len(dates)} days, picks_dirs={dirs}")
    print(f"  news filter at trial: enabled={tgc.NEWS_FILTER_ENABLED} "
          f"min={tgc.NEWS_MIN_ARTICLES} catalyst={tgc.NEWS_REQUIRE_CATALYST}")

    cash = STARTING_CASH
    daily = []
    for d in dates:
        day_picks = picks_by_date.get(d, [])
        if day_picks and tgc.NEWS_FILTER_ENABLED and (tgc.NEWS_MIN_ARTICLES > 0 or tgc.NEWS_REQUIRE_CATALYST):
            from news_filter import filter_picks as nfp
            day_picks = nfp(day_picks, d, tgc.NEWS_MIN_ARTICLES, tgc.NEWS_REQUIRE_CATALYST)
        if not day_picks:
            daily.append({"date": d, "pnl": 0, "equity": cash, "n_picks": 0})
            continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            _, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash
            )
        except Exception as e:
            daily.append({"date": d, "pnl": 0, "equity": cash, "err": str(e)[:80]})
            continue
        pnl = end_c - cash
        cash = end_c + (unset if is_cash else 0)
        daily.append({"date": d, "pnl": pnl, "equity": cash, "n_picks": len(day_picks)})
    return cash, daily


def main():
    os.makedirs("config", exist_ok=True)
    os.makedirs(OUTDIR, exist_ok=True)
    num, score, ua, params = fetch_best_params()
    print(f"=== Current best W7 trial: #{num} ===")
    print(f"  score = ${score:,.0f}")
    print(f"  user_attrs: total_pnl={ua.get('total_pnl','?')} "
          f"pf={ua.get('pf','?')} n={ua.get('n','?')} wr={ua.get('wr','?')}")
    print(f"  news filter setting: "
          f"enabled={params.get('enable_news_filter','?')} "
          f"min={params.get('min_news_articles','?')} "
          f"catalyst={params.get('require_news_catalyst','?')}")

    cfg_path = f"config/trial_{num}_w7_extracted.json"
    with open(cfg_path, "w") as f:
        json.dump({
            "trial_number": num,
            "score": float(score),
            "user_attrs": ua,
            "params": params,
        }, f, indent=2)
    print(f"  wrote {cfg_path}")

    print(f"\n=== Forward test #{num} on 2026 ===")
    final_cash, daily = forward_2026(params, f"#{num}")

    total_pnl = final_cash - STARTING_CASH
    wins = sum(1 for r in daily if r["pnl"] > 0)
    losses = sum(1 for r in daily if r["pnl"] < 0)
    flats = sum(1 for r in daily if r["pnl"] == 0)
    print(f"\n  Final equity:   ${final_cash:,.0f}")
    print(f"  Total PnL:      ${total_pnl:+,.0f}  ({final_cash/STARTING_CASH:.2f}x)")
    print(f"  Days w wins/losses/flat: {wins}/{losses}/{flats} of {len(daily)}")
    best = sorted(daily, key=lambda r: -r["pnl"])[:5]
    worst = sorted(daily, key=lambda r: r["pnl"])[:5]
    print(f"\n  Best 5 days:")
    for r in best:
        print(f"    {r['date']}  ${r['pnl']:+,.0f}")
    print(f"  Worst 5 days:")
    for r in worst:
        print(f"    {r['date']}  ${r['pnl']:+,.0f}")

    out_json = f"{OUTDIR}/W5_train_2022..2025_test_2026_PARTIAL_trial_{num}.json"
    with open(out_json, "w") as f:
        json.dump({
            "trial_number": num,
            "train_score": float(score),
            "user_attrs": ua,
            "forward_final_equity": final_cash,
            "forward_total_pnl": total_pnl,
            "wins": wins,
            "losses": losses,
            "daily": daily,
        }, f, indent=2)
    print(f"\n  wrote {out_json}")


if __name__ == "__main__":
    main()
