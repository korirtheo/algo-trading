"""Forward test #511 vs #561 vs W13 baseline on 2022 OOS data.

2022 is COMPLETELY untouched during training of any W21x study.
2022 had real microcap pump activity (avg 6.9 picks/day, full 251 days),
unlike 2023 which was thin (0.9 picks/day, incomplete data download).

If #561 still beats #511 on 2022 -> deploy swap is justified.
If #561 loses on 2022 -> the 2026 Mar-Jun edge is regime-specific, stay with #511.
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
W13_DEPLOY = "config/trial_w13_1202_deploy.json"
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
DATA_DIRS = ["stored_data_2022"]
OUTDIR = "results"
ALL_STRATS = ["h", "g", "a", "f", "d", "v", "p", "m", "r", "w", "o", "b", "k", "c", "s", "e", "i", "j", "n", "l", "x"]
EXIT_PARAMS = ["g_target_pct", "g_stop_pct", "g_time_limit_min", "g_trail_pct", "g_trail_activate_pct"]


def fetch_basin_trials():
    c = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres", dbname="optuna_w21b")
    cur = c.cursor()
    cur.execute("""
        WITH p AS (
          SELECT t.trial_id, t.number, tv.value,
            MAX(CASE WHEN tp.param_name='g_target_pct' THEN tp.param_value END) tgt,
            MAX(CASE WHEN tp.param_name='g_stop_pct' THEN tp.param_value END) stp,
            MAX(CASE WHEN tp.param_name='g_time_limit_min' THEN tp.param_value END) tlim,
            MAX(CASE WHEN tp.param_name='g_trail_pct' THEN tp.param_value END) trl,
            MAX(CASE WHEN tp.param_name='g_trail_activate_pct' THEN tp.param_value END) tract
          FROM trials t JOIN trial_values tv ON tv.trial_id=t.trial_id
          LEFT JOIN trial_params tp ON tp.trial_id=t.trial_id
          WHERE t.state='COMPLETE' AND t.study_id=(SELECT study_id FROM studies WHERE study_name='w21b_target100')
          GROUP BY t.trial_id, t.number, tv.value
        )
        SELECT trial_id, number, value, tgt, stp, tlim, trl, tract FROM p
        WHERE tgt BETWEEN 60 AND 68 AND stp BETWEEN 22 AND 26
          AND tlim BETWEEN 7 AND 13 AND trl BETWEEN 0.4 AND 0.6
          AND tract <= 1
        ORDER BY value DESC
    """)
    rows = cur.fetchall()
    trials = []
    for tid, num, score, tgt, stp, tlim, trl, tract in rows:
        trials.append({
            "trial_id": tid, "number": num, "score": float(score),
            "params": {
                "g_target_pct": float(tgt), "g_stop_pct": float(stp),
                "g_time_limit_min": int(tlim), "g_trail_pct": float(trl),
                "g_trail_activate_pct": float(tract),
            }
        })
    c.close()
    return trials


def forward(args):
    exit_params, label = args
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD

    with open(BASELINE) as f:
        baseline = json.load(f)
    with open(W13_DEPLOY) as f:
        w13 = json.load(f)["params"]
    merged = {**baseline, **w13, **exit_params}
    for s in ALL_STRATS:
        merged[f"enable_{s}"] = (s in {"g", "l"})
    set_strategy_params(merged)

    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.MARGIN_MULTIPLIER = 1.0
    tgc.NEWS_MODULATOR_ENABLED = False

    dirs = [d for d in DATA_DIRS if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted(all_dates)

    cash = STARTING_CASH
    eq_curve = [cash]
    n_trades = 0
    g_n, g_w, g_pnl = 0, 0, 0
    l_n, l_w, l_pnl = 0, 0, 0
    for d in dates:
        day_picks = picks_by_date.get(d, [])
        if not day_picks:
            eq_curve.append(cash); continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(day_picks, cash, cash_account=is_cash)
        except Exception:
            eq_curve.append(cash); continue
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                n_trades += 1
                if st.get("strategy") == "G":
                    g_n += 1; g_pnl += st["pnl"]
                    if st["pnl"] > 0: g_w += 1
                elif st.get("strategy") == "L":
                    l_n += 1; l_pnl += st["pnl"]
                    if st["pnl"] > 0: l_w += 1
        cash = end_c + (unset if is_cash else 0)
        eq_curve.append(cash)
    eq = np.array(eq_curve)
    peak = np.maximum.accumulate(eq)
    dd_arr = (eq - peak) / peak * 100
    return {
        "label": label,
        "final": float(eq[-1]),
        "forward_pnl": float(eq[-1] - STARTING_CASH),
        "max_dd_pct": float(dd_arr.min()),
        "n_trades": n_trades,
        "n_days": len(dates),
        "g_n": g_n, "g_wr": g_w / g_n * 100 if g_n else 0, "g_pnl": g_pnl,
        "l_n": l_n, "l_wr": l_w / l_n * 100 if l_n else 0, "l_pnl": l_pnl,
    }


def main():
    print("=" * 92)
    print(f"  2022 BLIND OOS — W13 vs #511 vs #561 micro-basin")
    print(f"  Data: stored_data_2022 (~251 days, 6.9 picks/day)")
    print(f"  2022 has NEVER been in any W21x training window.")
    print("=" * 92)

    trials = fetch_basin_trials()
    print(f"  Micro-basin trials: {len(trials)}")

    with open(W13_DEPLOY) as f:
        w13_full = json.load(f)["params"]
    with open(W21B_DEPLOY) as f:
        w21b_full = json.load(f)["params"]
    w13_exits = {k: w13_full[k] for k in EXIT_PARAMS}
    w21b511_exits = {k: w21b_full[k] for k in EXIT_PARAMS}

    jobs = [
        (w13_exits, "W13_#1202_baseline"),
        (w21b511_exits, "W21b_#511_deployed"),
    ]
    for t in trials:
        jobs.append((t["params"], f"#{t['number']}"))

    print(f"  Total forward jobs: {len(jobs)}")
    print(f"  Running 6 parallel workers...")

    forwards = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(forward, j): j[1] for j in jobs}
        done = 0
        for fut in as_completed(futs):
            try:
                r = fut.result()
                num_str = r["label"].lstrip("#")
                if num_str.isdigit():
                    num = int(num_str)
                    t = next((x for x in trials if x["number"] == num), None)
                    if t:
                        r["trial_number"] = num
                        r["train_score"] = t["score"]
                        r["params"] = t["params"]
                forwards.append(r)
                done += 1
                p_str = ""
                if "params" in r:
                    p = r["params"]
                    p_str = f"  tgt={p['g_target_pct']:>4} time={p['g_time_limit_min']:>3}"
                print(f"    [{done:>2}/{len(jobs)}] {r['label']:<24} fwd ${r['forward_pnl']:>+10,.0f}  DD {r['max_dd_pct']:>6.1f}%  G:{r['g_n']:>3}/WR{r['g_wr']:>4.0f}% L:{r['l_n']:>3}/WR{r['l_wr']:>4.0f}%{p_str}")
            except Exception as e:
                print(f"    failed [{futs[fut]}]: {e}")

    base_w13 = next(r for r in forwards if r["label"] == "W13_#1202_baseline")
    base_511 = next(r for r in forwards if r["label"] == "W21b_#511_deployed")
    basin = [r for r in forwards if "trial_number" in r]
    basin.sort(key=lambda x: -x["forward_pnl"])
    time9 = [r for r in basin if r["params"]["g_time_limit_min"] == 9]
    time12 = [r for r in basin if r["params"]["g_time_limit_min"] == 12]

    print()
    print("=" * 92)
    print("  RESULTS — 2022 BLIND OOS")
    print("=" * 92)
    print(f"  W13 baseline:    fwd ${base_w13['forward_pnl']:>+10,.0f}  DD {base_w13['max_dd_pct']:>5.1f}%  G:{base_w13['g_n']}/WR{base_w13['g_wr']:.0f}% L:{base_w13['l_n']}/WR{base_w13['l_wr']:.0f}%")
    print(f"  #511 deployed:   fwd ${base_511['forward_pnl']:>+10,.0f}  DD {base_511['max_dd_pct']:>5.1f}%  G:{base_511['g_n']}/WR{base_511['g_wr']:.0f}% L:{base_511['l_n']}/WR{base_511['l_wr']:.0f}%")
    print()
    print(f"  time=9  trials (n={len(time9)}):   median fwd ${float(np.median([r['forward_pnl'] for r in time9])):>+10,.0f}")
    print(f"  time=12 trials (n={len(time12)}):   median fwd ${float(np.median([r['forward_pnl'] for r in time12])):>+10,.0f}")
    print()
    pct_beat_511 = sum(1 for r in basin if r['forward_pnl'] > base_511['forward_pnl']) / len(basin) * 100
    print(f"  % basin beating #511: {pct_beat_511:.1f}%")
    print()
    print(f"  Top 5 forward winners:")
    for i, r in enumerate(basin[:5]):
        p = r["params"]
        print(f"    {i+1}. #{r['trial_number']:<5} fwd ${r['forward_pnl']:>+9,.0f}  tgt={p['g_target_pct']} stop={p['g_stop_pct']} time={p['g_time_limit_min']} trail={p['g_trail_pct']}")

    summary = {
        "test": "2022 blind OOS — W13 vs #511 vs micro-basin",
        "baselines": {"W13_#1202": base_w13, "W21b_#511": base_511},
        "stats": {
            "pct_beat_511": pct_beat_511,
            "time9_median": float(np.median([r["forward_pnl"] for r in time9])),
            "time12_median": float(np.median([r["forward_pnl"] for r in time12])),
            "511_pnl": base_511["forward_pnl"],
        },
        "forwards": basin,
    }
    out_path = f"{OUTDIR}/forward_2022_w21b_561_vs_511.json"
    os.makedirs(OUTDIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()
