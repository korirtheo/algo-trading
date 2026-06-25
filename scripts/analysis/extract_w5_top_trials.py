"""Extract top W5 trials from the SQLite DB and save as config JSON files.

For each top trial, save:
  - trial_number
  - score
  - user_attrs (PF, sharpe, total_pnl, enabled strategies, etc.)
  - params (the full Optuna param dict)
  - metadata (study name, train years, test year)

Files written to config/trial_XXX_w5_extracted.json so they can be
loaded by validate_*_multiyear.py and other downstream scripts.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import sqlite3
import json
import os

DB = "results/w5_phase1a/W5_train_2022_2023_2024_2025_test_2026.db"
TOP_N = 6  # extract top 6 by score
OUT_DIR = "config"


def extract_trial(trial_id):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # Score
    cur.execute("SELECT value FROM trial_values WHERE trial_id=?", (trial_id,))
    score = cur.fetchone()[0]

    # Params — Optuna stores them in trial_params with name+value
    cur.execute("""SELECT param_name, param_value, distribution_json
                   FROM trial_params WHERE trial_id=?""", (trial_id,))
    params = {}
    for name, value, dist_json in cur.fetchall():
        # Distributions can be int/float/categorical; we need the actual typed value
        try:
            dist = json.loads(dist_json)
        except Exception:
            dist = {}
        dist_name = dist.get("name", "")
        if dist_name == "IntDistribution":
            params[name] = int(value)
        elif dist_name == "CategoricalDistribution":
            choices = dist.get("attributes", {}).get("choices", [])
            if choices and int(value) < len(choices):
                params[name] = choices[int(value)]
            else:
                params[name] = value
        else:
            params[name] = float(value)

    # User attrs
    cur.execute("SELECT key, value_json FROM trial_user_attributes WHERE trial_id=?", (trial_id,))
    user_attrs = {}
    for k, v in cur.fetchall():
        try:
            user_attrs[k] = json.loads(v)
        except Exception:
            user_attrs[k] = v

    return {
        "trial_number": trial_id,
        "score": score,
        "study": "W5_train_2022_2023_2024_2025_test_2026",
        "tag": "wf_pf_microcap_pump_phase1a_W5",
        "objective": "PF (legacy total_pnl * min(pf, 3.0))",
        "slippage_model": "multiwindow + vol-adj, K=3",
        "shape_filter": "microcap-thin,thin-microcap",
        "X_strategy": "DISABLED (X_MIN_FIRST_LEG_GAIN_PCT=9999)",
        "min_trades_constraint": 100,
        "phase_1a_enabled": True,
        "train_years": ["2022", "2023", "2024", "2025"],
        "test_year": "2026",
        "user_attrs": user_attrs,
        "params": params,
    }


def main():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""SELECT t.trial_id, tv.value
                   FROM trials t JOIN trial_values tv ON t.trial_id=tv.trial_id
                   WHERE t.state='COMPLETE'
                   ORDER BY tv.value DESC LIMIT ?""", (TOP_N,))
    top = cur.fetchall()

    print(f"Extracting top {len(top)} W5 trials\n")
    for tid, score in top:
        data = extract_trial(tid)
        out = os.path.join(OUT_DIR, f"trial_{tid}_w5_extracted.json")
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        attrs = data['user_attrs']
        print(f"  #{tid}  score=${score:,.0f}  PnL=${attrs.get('total_pnl',0):,.0f}  "
              f"PF={attrs.get('pf','?')}  enabled={attrs.get('enabled','?')}")
        print(f"    -> {out}")


if __name__ == "__main__":
    main()
