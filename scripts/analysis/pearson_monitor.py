"""On-demand Pearson health check for a running Optuna study.

Computes train_pnl <-> val_pnl correlation across all completed trials.
Use to detect overfit BEFORE wide-forward / before declaring a winner.

Usage:
  python scripts/analysis/pearson_monitor.py <db_url> <study_name>

Example:
  python scripts/analysis/pearson_monitor.py postgresql://postgres@127.0.0.1:5432/optuna_w17 w17_gl_multi_val
"""
import sys, json
import numpy as np
import psycopg2


def main():
    if len(sys.argv) < 3:
        print("Usage: pearson_monitor.py <postgres_url> <study_name>")
        sys.exit(1)
    db_url = sys.argv[1]
    study_name = sys.argv[2]

    # Parse url postgresql://user@host:port/dbname
    from urllib.parse import urlparse
    u = urlparse(db_url)
    conn = psycopg2.connect(host=u.hostname or "127.0.0.1", port=u.port or 5432,
                            user=u.username or "postgres", dbname=u.path.lstrip("/"))
    cur = conn.cursor()
    cur.execute("SELECT study_id FROM studies WHERE study_name=%s", (study_name,))
    row = cur.fetchone()
    if not row:
        print(f"Study '{study_name}' not found.")
        sys.exit(1)
    study_id = row[0]

    cur.execute("""SELECT t.trial_id, t.number, tv.value FROM trials t
                   LEFT JOIN trial_values tv ON tv.trial_id=t.trial_id
                   WHERE t.study_id=%s AND t.state='COMPLETE'
                   ORDER BY t.number""", (study_id,))
    rows = cur.fetchall()
    print(f"Study: {study_name}")
    print(f"Completed trials: {len(rows)}")

    train_vals = []; val_vals = []; val_window_keys = set()
    val1_vals = []; val2_vals = []; val3_vals = []
    score_vals = []
    for tid, num, score in rows:
        cur.execute("""SELECT key, value_json FROM trial_user_attributes
                       WHERE trial_id=%s
                       AND key IN ('train_pnl','val_pnl','val1_pnl','val2_pnl','val3_pnl','min_val_score')""", (tid,))
        attrs = {k: json.loads(v) if v else None for k, v in cur.fetchall()}
        tr = attrs.get("train_pnl")
        vl = attrs.get("val_pnl") or attrs.get("val1_pnl")
        if tr is None or vl is None: continue
        train_vals.append(float(tr))
        val_vals.append(float(vl))
        if attrs.get("val1_pnl") is not None: val1_vals.append(float(attrs["val1_pnl"]))
        if attrs.get("val2_pnl") is not None: val2_vals.append(float(attrs["val2_pnl"]))
        if attrs.get("val3_pnl") is not None: val3_vals.append(float(attrs["val3_pnl"]))
        if score is not None: score_vals.append(float(score))

    if len(train_vals) < 10:
        print(f"Too few train/val pairs ({len(train_vals)}). Need >=10. Wait for more trials.")
        return

    def pearson(a, b):
        if not a or not b or len(a) != len(b): return None
        return float(np.corrcoef(a, b)[0, 1])

    r_train_val = pearson(train_vals, val_vals)
    print(f"\n=== Pearson on {len(train_vals)} trials ===")
    print(f"train_pnl <-> val_pnl     : {r_train_val:+.4f}", _badge(r_train_val))

    if val1_vals and val2_vals:
        r_v1v2 = pearson(val1_vals[:len(val2_vals)], val2_vals[:len(val1_vals)])
        if r_v1v2 is not None:
            print(f"val1_pnl  <-> val2_pnl    : {r_v1v2:+.4f}", _badge(r_v1v2))
    if val1_vals and val3_vals:
        r_v1v3 = pearson(val1_vals[:len(val3_vals)], val3_vals[:len(val1_vals)])
        if r_v1v3 is not None:
            print(f"val1_pnl  <-> val3_pnl    : {r_v1v3:+.4f}", _badge(r_v1v3))
    if val2_vals and val3_vals:
        r_v2v3 = pearson(val2_vals[:len(val3_vals)], val3_vals[:len(val2_vals)])
        if r_v2v3 is not None:
            print(f"val2_pnl  <-> val3_pnl    : {r_v2v3:+.4f}", _badge(r_v2v3))

    # Collect val-pair Pearsons for regime-shift detection
    val_pair_pearsons = []
    if val1_vals and val2_vals:
        n = min(len(val1_vals), len(val2_vals))
        r = pearson(val1_vals[:n], val2_vals[:n])
        if r is not None: val_pair_pearsons.append(("val1-vs-val2", r))
    if val1_vals and val3_vals:
        n = min(len(val1_vals), len(val3_vals))
        r = pearson(val1_vals[:n], val3_vals[:n])
        if r is not None: val_pair_pearsons.append(("val1-vs-val3", r))
    if val2_vals and val3_vals:
        n = min(len(val2_vals), len(val3_vals))
        r = pearson(val2_vals[:n], val3_vals[:n])
        if r is not None: val_pair_pearsons.append(("val2-vs-val3", r))

    print(f"\n=== Interpretation ===")
    print(f"  train_pnl <-> val_pnl is the WEAK overfit detector — only catches")
    print(f"  trials that memorize training. CANNOT detect regime-overfit when val")
    print(f"  is too close in regime-space to training (W16 had +0.86 here but")
    print(f"  blew up on blind OOS).")
    if r_train_val > 0.3:
        print(f"  train<->val: OK (no training-memorization overfit detected).")
    else:
        print(f"  train<->val: WARN — training may not generalize even to val.")

    if val_pair_pearsons:
        weakest = min(val_pair_pearsons, key=lambda x: x[1])
        print(f"\n  REGIME consistency (val<->val across windows):")
        for label, r in val_pair_pearsons:
            badge = "OK" if r > 0.3 else ("MARGINAL" if r > 0.1 else "REGIME-DIVERGENT")
            print(f"    {label}: {r:+.3f}  [{badge}]")
        print(f"  Weakest pair: {weakest[0]} = {weakest[1]:+.3f}")
        if weakest[1] < 0.1:
            print(f"  **REGIME-OVERFIT WARN** — trial that wins {weakest[0].split('-vs-')[0]} "
                  f"doesn't win {weakest[0].split('-vs-')[1]}. min(val_score) objective should "
                  f"penalize this, but the basin may have no fully-generalizing config.")
        elif weakest[1] < 0.3:
            print(f"  MARGINAL — regime spread is detectable but TPE may still find a robust basin.")
        else:
            print(f"  OK — trial rankings are consistent across val regimes.")
    else:
        print(f"\n  NO multi-window val data — can't measure regime consistency.")
        print(f"  This study has the W16-style weakness: train<->val correlation alone")
        print(f"  doesn't catch regime-shift overfit. Re-run with --val-windows for diagnosis.")
    conn.close()


def _badge(r):
    if r is None: return ""
    if r > 0.3: return "[OK]"
    if r > -0.1: return "[FLAT]"
    if r > -0.3: return "[WARN]"
    return "[**OVERFIT**]"


if __name__ == "__main__":
    main()
