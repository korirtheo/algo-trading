"""Monitor the gl_1min_v1 optuna study: report progress + best trial.

Usage: python monitor_gl1min.py [--interval 300] [--max 12]
"""
import argparse
import sys
import time

import optuna

DB = "postgresql://postgres@127.0.0.1:5432/optuna_gl_1min"
STUDY = "gl_1min_v1"
TARGET = 1500


def snapshot():
    s = optuna.load_study(study_name=STUDY, storage=DB)
    comp = [t for t in s.trials if t.state.name == "COMPLETE"]
    n = len(comp)
    pct = n / TARGET * 100
    line = f"[{n}/{TARGET}] {pct:.0f}%"
    if comp:
        best = s.best_trial
        ua = best.user_attrs
        g = ua.get("enable_g"); l = ua.get("enable_l")
        line += (f" | best #{best.number} score={best.value:,.0f} "
                 f"PnL=${ua.get('total_pnl',0):,.0f} PF={ua.get('pf',0):.2f} "
                 f"WR={ua.get('wr',0):.1f}% n={ua.get('n',0)} "
                 f"[G={g} L={l}]")
    else:
        line += " | no completed trials yet"
    return line


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=300.0)
    p.add_argument("--max", type=int, default=0, help="max samples (0 = until done)")
    args = p.parse_args()

    done = 0
    while True:
        try:
            line = snapshot()
            print(time.strftime("%H:%M:%S"), line, flush=True)
        except Exception as e:
            print(time.strftime("%H:%M:%S"), "ERR:", str(e)[:80], flush=True)
        done += 1
        if args.max and done >= args.max:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
