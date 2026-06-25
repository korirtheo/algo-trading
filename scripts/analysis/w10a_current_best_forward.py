"""Forward test the current best W10a trial on 2026."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
import psycopg2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
DATA_DIRS_2026 = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]
OUTDIR = "results/w10a_forward"


def fetch_top_trial(db="optuna_w10"):
    c = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres", dbname=db)
    cur = c.cursor()
    cur.execute("""SELECT t.trial_id, t.number, tv.value
                   FROM trials t JOIN trial_values tv ON tv.trial_id=t.trial_id
                   WHERE t.state='COMPLETE' ORDER BY tv.value DESC LIMIT 1""")
    tid, num, score = cur.fetchone()
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
    return {"number": num, "score": float(score), "user_attrs": ua, "params": params}


def forward(trial):
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD

    with open(BASELINE) as f: baseline = json.load(f)
    merged = dict(baseline); merged.update(trial["params"])

    # FORCE_ENABLE_STRATS=g,l for W10
    ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]
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

    dirs = [d for d in DATA_DIRS_2026 if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if d.startswith("2026")])

    cash = STARTING_CASH
    daily_eq = [cash]
    trades = []
    daily_records = []
    for d in dates:
        day_picks = picks_by_date.get(d, [])
        if not day_picks:
            daily_eq.append(cash)
            daily_records.append({"date": d, "pnl": 0, "equity": cash, "n_trades": 0})
            continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception:
            daily_eq.append(cash)
            daily_records.append({"date": d, "pnl": 0, "equity": cash, "n_trades": 0})
            continue
        day_pnl = 0
        n_today = 0
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                day_pnl += st["pnl"]
                n_today += 1
                trades.append({
                    "date": d,
                    "ticker": st["ticker"],
                    "strategy": st.get("strategy", "?"),
                    "pnl": float(st["pnl"]),
                    "cost": float(st["position_cost"]),
                })
        cash = end_c + (unset if is_cash else 0)
        daily_eq.append(cash)
        daily_records.append({"date": d, "pnl": day_pnl, "equity": cash, "n_trades": n_today})

    return cash, daily_eq, trades, daily_records, dates


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("Fetching current best W10a trial from Postgres...")
    trial = fetch_top_trial()
    p = trial["params"]
    ua = trial["user_attrs"]
    print(f"  Trial #{trial['number']} score=${trial['score']:,.0f}")
    print(f"  Train: pf={ua.get('pf','-')} n={ua.get('n','-')} wr={ua.get('wr','-')}")
    print(f"  g_min_gap_pct={p.get('g_min_gap_pct','-')}  l_min_gap={p.get('l_min_gap','-')}")

    # Save config
    cfg_path = f"config/trial_w10a_{trial['number']}_extracted.json"
    with open(cfg_path, "w") as f:
        json.dump({
            "trial_number": trial["number"],
            "score": trial["score"],
            "user_attrs": ua,
            "params": p,
        }, f, indent=2, default=str)
    print(f"  Saved {cfg_path}")

    print(f"\nForward-testing on 2026 (G+L only, current execution model)...")
    final_cash, daily_eq, trades, daily_records, dates = forward(trial)
    pnl = final_cash - STARTING_CASH
    multi = final_cash / STARTING_CASH

    # Per-strategy attribution
    by_strat = defaultdict(lambda: {"n": 0, "pnl": 0.0})
    for t in trades:
        by_strat[t["strategy"]]["n"] += 1
        by_strat[t["strategy"]]["pnl"] += t["pnl"]

    print(f"\n=== FORWARD 2026 RESULT ===")
    print(f"  Final equity: ${final_cash:,.0f}  ({multi:.2f}x)")
    print(f"  Total PnL:    ${pnl:+,.0f}")
    print(f"  Trades:       {len(trades)}")
    wins = sum(1 for t in trades if t["pnl"] > 0)
    print(f"  Win rate:     {wins}/{len(trades)} = {wins/max(len(trades),1)*100:.1f}%")
    eq_arr = np.array(daily_eq)
    peak = np.maximum.accumulate(eq_arr)
    dd_pct = (eq_arr - peak) / peak * 100 if peak.max() > 0 else 0
    print(f"  Max DD:       {dd_pct.min():.1f}%  (${(eq_arr-peak).min():,.0f})")
    print(f"\n  Per-strategy:")
    for s, v in sorted(by_strat.items()):
        avg = v["pnl"] / v["n"] if v["n"] else 0
        print(f"    {s:<2}  {v['n']:>3} trades  ${v['pnl']:>+11,.0f}  (${avg:+,.0f}/trade)")

    # Compare to baselines
    print(f"\n=== vs BASELINES ===")
    baselines = [
        ("#124 W3 deployed",        22_534),
        ("W7 #254 (forward winner)", 199_553),
        ("W8 #1550 (CV-min best)",   36_668),
        ("W9 #1091 (geomean best)",  87_642),
        ("#254 -L (ablation)",       254_167),
        ("G+V+L composition",        225_073),  # roughly
    ]
    print(f"  {'config':<35} {'fwd PnL':>12} {'delta vs W10a':>16}")
    print(f"  {'-'*35} {'-'*12} {'-'*16}")
    for label, baseline_pnl in baselines:
        delta = pnl - baseline_pnl
        marker = "✓" if delta > 0 else "✗"
        print(f"  {label:<35} ${baseline_pnl:>+10,.0f}  ${delta:>+12,.0f} {marker}")

    # Save results
    out = {
        "trial_number": trial["number"],
        "train_score": trial["score"],
        "train_attrs": ua,
        "g_min_gap_pct": p.get("g_min_gap_pct"),
        "l_min_gap": p.get("l_min_gap"),
        "forward_final": float(final_cash),
        "forward_pnl": float(pnl),
        "forward_multi": float(multi),
        "forward_max_dd_pct": float(dd_pct.min()),
        "forward_n_trades": len(trades),
        "forward_win_rate": wins / max(len(trades), 1) * 100,
        "per_strategy": {k: {"n": v["n"], "pnl": v["pnl"]} for k, v in by_strat.items()},
        "daily_records": daily_records,
    }
    out_path = f"{OUTDIR}/forward_trial_{trial['number']}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved {out_path}")

    # Equity curve chart
    dt = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
    dt = [dt[0]] + dt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(dt, eq_arr, color="#2ca02c", linewidth=2.5,
              label=f"W10a #{trial['number']} (g={p.get('g_min_gap_pct')}%, l={p.get('l_min_gap')}%)")
    ax1.axhline(STARTING_CASH, color="black", linestyle=":", alpha=0.5)
    # baseline lines
    ax1.axhline(STARTING_CASH + 22_534, color="#888888", linestyle="--", alpha=0.6, label="#124 deployed")
    ax1.axhline(STARTING_CASH + 199_553, color="#1f77b4", linestyle="--", alpha=0.6, label="W7 #254")
    ax1.axhline(STARTING_CASH + 254_167, color="#d62728", linestyle="--", alpha=0.6, label="#254 -L")
    ax1.set_ylabel("Equity ($)")
    ax1.set_title(f"W10a current best (#{trial['number']}) forward on 2026 — G+L only, no shape filter\n"
                  f"Train score ${trial['score']/1e6:.1f}M  →  Forward 2026 ${pnl:+,.0f}")
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax1.grid(True, alpha=0.3)
    ax2.fill_between(dt, dd_pct, 0, color="#d62728", alpha=0.3)
    ax2.plot(dt, dd_pct, color="#d62728", linewidth=1.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("2026 trading day")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout()
    chart_path = f"{OUTDIR}/equity_curve_trial_{trial['number']}.png"
    fig.savefig(chart_path, dpi=140); plt.close(fig)
    print(f"  Wrote {chart_path}")


if __name__ == "__main__":
    main()
