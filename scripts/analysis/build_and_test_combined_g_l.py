"""Build combined G+L config using G #159 (g_only_w21b) + L #626 (l_only_w21b),
then run compounded backtest on 2026 Mar-Jun vs #511 baseline.

Outputs:
  - config/trial_g_only_159_l_only_626_combined.json  (deploy-ready)
  - results/combined_g_l_vs_511_2026_mar_jun.json     (comparison data)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json, os
import numpy as np
import psycopg2

STARTING_CASH = 25_000
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
BASELINE = "config/trial_432_params.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]

WINDOWS = {
    "2022":         {"dirs": ["stored_data_2022"],
                    "lo": "2022-01-01", "hi": "2022-12-31"},
    "2026_MarJun":  {"dirs": ["stored_data_mar_may_2026", "stored_data_jun_2026"],
                    "lo": "2026-03-01", "hi": "2026-06-30"},
}


def fetch_trial_params(db, study, trial_num):
    c = psycopg2.connect(host='127.0.0.1', port=5432, user='postgres', dbname=db)
    cur = c.cursor()
    cur.execute('SELECT trial_id FROM trials WHERE number=%s AND study_id=(SELECT study_id FROM studies WHERE study_name=%s)', (trial_num, study))
    tid = cur.fetchone()[0]
    cur.execute('SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=%s', (tid,))
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
            params[name] = float(val)
    c.close()
    return params


def build_combined():
    """Build combined config: #511 base + G #159's g_* params + L #626's l_* params."""
    with open(W21B_DEPLOY) as f: base = json.load(f)
    combined = dict(base["params"])
    g159 = fetch_trial_params("optuna_g_only", "g_only_w21b", 159)
    l626 = fetch_trial_params("optuna_l_only", "l_only_w21b", 626)
    # Override only g_* and l_* keys
    n_g_overrides = n_l_overrides = 0
    for k, v in g159.items():
        if k.startswith("g_"):
            combined[k] = v; n_g_overrides += 1
    for k, v in l626.items():
        if k.startswith("l_"):
            combined[k] = v; n_l_overrides += 1
    # Force enable_g, enable_l only
    for s in ALL_STRATS:
        combined[f"enable_{s}"] = (s in {"g", "l"})
    print(f"Built combined config: overrode {n_g_overrides} g_* and {n_l_overrides} l_* params")
    return combined


def run_compounded(params, label, dirs_arg, date_lo, date_hi):
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

    dirs = [d for d in dirs_arg if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if date_lo <= d <= date_hi])

    cash = STARTING_CASH
    trades_by_strat = {"G": [], "L": []}
    daily_eq = [cash]
    for d in dates:
        dp = picks_by_date.get(d, [])
        if not dp: daily_eq.append(cash); continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except Exception:
            daily_eq.append(cash); continue
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0:
                s = st.get("strategy")
                p = float(st.get("pnl") or 0)
                if s in trades_by_strat:
                    trades_by_strat[s].append({"date": d, "ticker": st.get("ticker"),
                                               "pnl": p, "reason": st.get("exit_reason")})
        cash = end_c + (unset if is_cash else 0)
        daily_eq.append(cash)
    eq = np.array(daily_eq)
    peak = np.maximum.accumulate(eq)
    dd_pct = float(((eq - peak) / peak).min() * 100) if len(peak) > 0 else 0.0

    out = {"label": label, "final_cash": float(cash), "pnl": float(cash - STARTING_CASH),
           "multiplier": cash/STARTING_CASH, "max_dd_pct": dd_pct,
           "n_g": len(trades_by_strat["G"]), "n_l": len(trades_by_strat["L"]),
           "g_pnl": sum(t["pnl"] for t in trades_by_strat["G"]),
           "l_pnl": sum(t["pnl"] for t in trades_by_strat["L"]),
           "g_wr": (sum(1 for t in trades_by_strat["G"] if t["pnl"]>0)/len(trades_by_strat["G"])*100) if trades_by_strat["G"] else 0,
           "l_wr": (sum(1 for t in trades_by_strat["L"] if t["pnl"]>0)/len(trades_by_strat["L"])*100) if trades_by_strat["L"] else 0,
           "trades": trades_by_strat}
    return out


def main():
    # Build combined
    combined = build_combined()
    out_cfg = "config/trial_g_only_159_l_only_626_combined.json"
    with open(out_cfg, "w") as f:
        json.dump({
            "label": "G #159 (g_only_w21b) + L #626 (l_only_w21b) combined",
            "source": "scripts/analysis/build_and_test_combined_g_l.py",
            "g_override_source": "optuna_g_only.g_only_w21b trial #159 (train rank 352, wide-fwd #1 at $1.5M)",
            "l_override_source": "optuna_l_only.l_only_w21b trial #626 (train rank 165, wide-fwd #1 at $408K)",
            "params": combined,
        }, f, indent=2, default=str)
    print(f"Wrote combined config: {out_cfg}")

    with open(W21B_DEPLOY) as f: base_data = json.load(f)
    with open(BASELINE) as f: bp = json.load(f)
    p511 = {**bp, **base_data["params"]}
    for s in ALL_STRATS:
        p511[f"enable_{s}"] = (s in {"g", "l"})

    all_results = {}
    for wname, w in WINDOWS.items():
        print(f"\n--- Running #511 baseline on {wname} ---")
        r511 = run_compounded(p511, "#511 baseline", w["dirs"], w["lo"], w["hi"])
        print(f"--- Running G #159 + L #626 combined on {wname} ---")
        full = {**bp, **combined}
        r_new = run_compounded(full, "G #159 + L #626", w["dirs"], w["lo"], w["hi"])
        all_results[wname] = {"baseline_511": r511, "combined_g_l": r_new}

        print(f"\n{'='*90}")
        print(f"  COMBINED G+L vs #511 on {wname} (compounded backtest)")
        print(f"{'='*90}")
        print(f"  {'metric':<22} {'#511 baseline':>18} {'G #159 + L #626':>20} {'Δ':>14}")
        for k, label in [("pnl", "Total PnL"), ("multiplier", "Multiplier"),
                          ("max_dd_pct", "Max DD %"),
                          ("n_g", "G trades"), ("g_pnl", "G PnL"), ("g_wr", "G WR%"),
                          ("n_l", "L trades"), ("l_pnl", "L PnL"), ("l_wr", "L WR%")]:
            v1 = r511[k]; v2 = r_new[k]
            if isinstance(v1, float): fmt1 = f"${v1:>+,.0f}" if "pnl" in k else f"{v1:.2f}"
            else: fmt1 = str(v1)
            if isinstance(v2, float): fmt2 = f"${v2:>+,.0f}" if "pnl" in k else f"{v2:.2f}"
            else: fmt2 = str(v2)
            try: delta = f"+{(v2-v1)/abs(v1)*100:.1f}%" if v1 != 0 else "n/a"
            except: delta = ""
            print(f"  {label:<22} {fmt1:>18} {fmt2:>20} {delta:>14}")

    out_json = "results/combined_g_l_vs_511_compounded.json"
    os.makedirs("results", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
