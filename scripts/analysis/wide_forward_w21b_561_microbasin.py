"""Focused wide-forward for W21b #561 micro-basin.

#561 was found at training rank 267 with target=64, stop=24, time=9, trail=0.5 —
beats #511 on Mar-Jun OOS by +8% ($686K vs $636K). Investigate the local basin
to confirm robustness.

Basin definition (#561's neighborhood):
  target_pct in [60, 68]
  stop_pct in [22, 26]
  time_limit_min in [7, 13]
  trail_pct in [0.4, 0.6]
  trail_activate_pct <= 1
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
DATA_DIRS = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]
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
    dates = sorted([d for d in all_dates if "2026-03-01" <= d <= "2026-06-30"])

    cash = STARTING_CASH
    eq_curve = [cash]
    n_trades = 0
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
    }


def main():
    print("=" * 92)
    print("  W21b #561 MICRO-BASIN focused wide-forward")
    print("  Basin: target [60-68], stop [22-26], time [7-13], trail [0.4-0.6], trail_act <= 1")
    print("  Forward window: 2026-03-01 -> 2026-06-30 (true OOS)")
    print("=" * 92)

    trials = fetch_basin_trials()
    print(f"  Trials in micro-basin: {len(trials)}")

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

    print(f"  Total forward jobs: {len(jobs)} (incl. 2 baselines)")
    print(f"  Running 6 parallel workers...")

    forwards = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(forward, j): j[1] for j in jobs}
        done = 0
        for fut in as_completed(futs):
            try:
                r = fut.result()
                # Attach trial meta if it's a basin trial
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
                    p_str = f"  tgt={p['g_target_pct']:>4} stp={p['g_stop_pct']:>4} time={p['g_time_limit_min']:>3} trl={p['g_trail_pct']:>3}"
                print(f"    [{done:>2}/{len(jobs)}] {r['label']:<24} fwd ${r['forward_pnl']:>+11,.0f}  DD {r['max_dd_pct']:>6.1f}%  n={r['n_trades']:>3}{p_str}")
            except Exception as e:
                print(f"    failed [{futs[fut]}]: {e}")

    base_w13 = next(r for r in forwards if r["label"] == "W13_#1202_baseline")
    base_511 = next(r for r in forwards if r["label"] == "W21b_#511_deployed")
    basin_rows = [r for r in forwards if "trial_number" in r]
    basin_rows.sort(key=lambda x: -x["forward_pnl"])

    xs = np.array([r["train_score"] for r in basin_rows])
    ys = np.array([r["forward_pnl"] for r in basin_rows])
    pearson = float(np.corrcoef(xs, ys)[0, 1]) if xs.std() > 0 else 0
    median = float(np.median(ys)); maxv = float(ys.max()); minv = float(ys.min())
    pct_beat_511 = sum(1 for v in ys if v > base_511["forward_pnl"]) / len(ys) * 100
    pct_beat_w13 = sum(1 for v in ys if v > base_w13["forward_pnl"]) / len(ys) * 100

    print()
    print("=" * 92)
    print("  RESULTS")
    print("=" * 92)
    print(f"  W13 baseline:    fwd ${base_w13['forward_pnl']:>+11,.0f}  DD {base_w13['max_dd_pct']:>6.1f}%")
    print(f"  #511 deployed:   fwd ${base_511['forward_pnl']:>+11,.0f}  DD {base_511['max_dd_pct']:>6.1f}%")
    print()
    print(f"  Micro-basin (n={len(basin_rows)}):")
    print(f"    median fwd:   ${median:>+11,.0f}")
    print(f"    max fwd:      ${maxv:>+11,.0f}")
    print(f"    min fwd:      ${minv:>+11,.0f}")
    print(f"  Pearson train -> forward: {pearson:+.3f}")
    print(f"  % beating W13:   {pct_beat_w13:.1f}%")
    print(f"  % beating #511:  {pct_beat_511:.1f}%")
    print()
    print(f"  Top 10 forward winners in micro-basin:")
    print(f"  {'rank':<4} {'trial':<6} {'fwd $':>11} {'DD%':>7} {'tgt':>4} {'stp':>4} {'time':>4} {'trl':>4} {'tact':>4}")
    for i, r in enumerate(basin_rows[:10]):
        p = r["params"]
        print(f"  {i+1:<4} #{r['trial_number']:<5} ${r['forward_pnl']:>+9,.0f} {r['max_dd_pct']:>6.1f}% {p['g_target_pct']:>4} {p['g_stop_pct']:>4} {p['g_time_limit_min']:>4} {p['g_trail_pct']:>4} {p['g_trail_activate_pct']:>4}")

    summary = {
        "study": "W21b #561 micro-basin",
        "forward_window": "2026-03-01 to 2026-06-30",
        "basin_definition": {
            "g_target_pct": [60, 68], "g_stop_pct": [22, 26],
            "g_time_limit_min": [7, 13], "g_trail_pct": [0.4, 0.6],
            "g_trail_activate_pct_max": 1,
        },
        "n_trials": len(basin_rows),
        "baselines": {"W13_#1202": base_w13, "W21b_#511": base_511},
        "stats": {
            "pearson_train_vs_forward": pearson,
            "median_forward": median, "max_forward": maxv, "min_forward": minv,
            "pct_beat_W13": pct_beat_w13, "pct_beat_511": pct_beat_511,
        },
        "forwards": basin_rows,
    }
    out_path = f"{OUTDIR}/wide_forward_w21b_561_microbasin.json"
    os.makedirs(OUTDIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()
