"""ChatGPT-style statistical analysis of wide_forward results across W7/W8/W9.

Reads results/wide_forward_w{7,8,9}.json (60 trials each = 30 top-by-train +
30 random middle) and computes:
  - Distribution stats: median, 90th pct, best, std
  - Train -> Forward correlation (Pearson + Spearman)
  - Top-10-by-train ∩ Top-10-by-forward hit rate
  - Top30 (by train) vs Random30 forward distributions
    (does the objective actually enrich for forward performance?)
"""
import json
import os
import statistics
import numpy as np

OUTDIR = "results"
STUDIES = ["w7", "w8", "w9"]


def load(study):
    path = f"{OUTDIR}/wide_forward_{study}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def percentile(xs, p):
    return float(np.percentile(xs, p))


def rank_correlation(xs, ys):
    def _rank(a):
        o = np.argsort(a); r = np.empty_like(o, dtype=float); r[o] = np.arange(len(a)); return r
    return float(np.corrcoef(_rank(xs), _rank(ys))[0, 1])


def main():
    print(f"\n{'='*92}")
    print(f"  WIDE-FORWARD DISTRIBUTION ANALYSIS  (per-study + cross-study)")
    print(f"{'='*92}")

    summary = {}
    for s in STUDIES:
        rows = load(s)
        if not rows:
            print(f"\n  {s.upper()}: no data")
            continue

        forward = np.array([r["forward_pnl"] for r in rows])
        train_score = np.array([r["train_score"] for r in rows])
        train_rank = np.array([r["train_rank"] for r in rows])

        # Distribution
        worst = float(forward.min())
        median = float(np.median(forward))
        p75 = percentile(forward, 75)
        p90 = percentile(forward, 90)
        best = float(forward.max())
        std = float(forward.std())
        win_rate = float((forward > 0).mean())

        # Correlations
        pearson = float(np.corrcoef(train_score, forward)[0, 1])
        spearman = rank_correlation(train_score, forward)
        # Rank-based: train_rank (1=best) vs forward_rank (1=best)
        forward_rank = train_rank.copy()
        order = np.argsort(-forward)  # descending
        forward_rank_arr = np.empty_like(order, dtype=float)
        forward_rank_arr[order] = np.arange(1, len(order) + 1)
        spearman_rank = float(np.corrcoef(train_rank, forward_rank_arr)[0, 1])

        # Top-decile hit rate: of trials with train_rank in top-10, how many forward in top-10?
        top10_train_mask = train_rank <= 10
        top10_forward_mask = forward_rank_arr <= 10
        hit = int((top10_train_mask & top10_forward_mask).sum())

        # Top30 (by train, train_rank <= 30) vs Random30 (train_rank > 30)
        top30_fwd = forward[train_rank <= 30]
        mid_fwd = forward[train_rank > 30]
        top30_median = float(np.median(top30_fwd)) if len(top30_fwd) else None
        mid_median = float(np.median(mid_fwd)) if len(mid_fwd) else None
        top30_p90 = percentile(top30_fwd, 90) if len(top30_fwd) else None
        mid_p90 = percentile(mid_fwd, 90) if len(mid_fwd) else None
        enrichment_median = (top30_median - mid_median) if (top30_median is not None and mid_median is not None) else None

        # Best-trial details
        best_idx = int(forward.argmax())
        best_trial = rows[best_idx]

        # Top-5 trials by forward (with their train rank)
        sorted_idx = np.argsort(-forward)
        top5_fwd = [rows[i] for i in sorted_idx[:5]]
        top5_train_ranks = [t["train_rank"] for t in top5_fwd]

        summary[s] = {
            "n": len(rows),
            "worst": worst, "median": median, "p75": p75, "p90": p90,
            "best": best, "std": std, "win_rate": win_rate,
            "pearson_train_fwd": pearson,
            "spearman_train_fwd": spearman,
            "spearman_rank_rank": spearman_rank,
            "top10_hit_rate": hit,
            "top30_fwd_median": top30_median,
            "mid_fwd_median": mid_median,
            "top30_fwd_p90": top30_p90,
            "mid_fwd_p90": mid_p90,
            "enrichment_median": enrichment_median,
            "best_trial": best_trial["trial_number"],
            "best_train_rank": best_trial["train_rank"],
            "best_forward_pnl": best,
            "top5_by_fwd_train_ranks": top5_train_ranks,
        }

        print(f"\n{'='*92}")
        print(f"  {s.upper()}  ({len(rows)} trials forward-tested)")
        print(f"{'='*92}")
        print(f"\n  FORWARD PnL DISTRIBUTION:")
        print(f"    worst    = ${worst:>+12,.0f}")
        print(f"    median   = ${median:>+12,.0f}")
        print(f"    p75      = ${p75:>+12,.0f}")
        print(f"    p90      = ${p90:>+12,.0f}")
        print(f"    best     = ${best:>+12,.0f}   (trial #{best_trial['trial_number']}, train_rank {best_trial['train_rank']})")
        print(f"    std      = ${std:>+12,.0f}")
        print(f"    win rate = {win_rate*100:.1f}% of trials forward > 0")

        print(f"\n  TRAIN -> FORWARD CORRELATION:")
        print(f"    Pearson(train_score, forward_pnl)  = {pearson:+.3f}")
        print(f"    Spearman(train_score, forward_pnl) = {spearman:+.3f}")
        print(f"    Spearman(train_rank, forward_rank) = {spearman_rank:+.3f}  (positive = bad: high train rank = low fwd rank)")

        print(f"\n  TOP-10 HIT RATE:")
        print(f"    Trials with train_rank in top-10 that ALSO end up in forward top-10: {hit}/10")

        print(f"\n  TOP-5 BY FORWARD — their train ranks: {top5_train_ranks}")
        print(f"    (If the objective ranks well, these should mostly be small numbers <= 30)")

        print(f"\n  ENRICHMENT (does top-by-train forward better than random-middle?):")
        if top30_median is not None and mid_median is not None:
            print(f"    Top30-by-train median forward:    ${top30_median:>+12,.0f}")
            print(f"    Random-middle median forward:     ${mid_median:>+12,.0f}")
            print(f"    Enrichment (top30 - random):      ${enrichment_median:>+12,.0f}")
            print(f"    Top30 p90:                        ${top30_p90:>+12,.0f}")
            print(f"    Random p90:                       ${mid_p90:>+12,.0f}")

    # Cross-study summary
    print(f"\n\n{'='*92}")
    print(f"  CROSS-STUDY COMPARISON — what matters for deployment")
    print(f"{'='*92}")
    print(f"\n  {'study':<5} {'median':>10} {'p90':>10} {'best':>12} {'std':>10} {'win%':>6}")
    for s in STUDIES:
        if s not in summary: continue
        x = summary[s]
        print(f"  {s.upper():<5} ${x['median']:>+8,.0f} ${x['p90']:>+8,.0f} ${x['best']:>+10,.0f} "
              f"${x['std']:>+8,.0f} {x['win_rate']*100:>5.1f}%")

    print(f"\n  {'study':<5} {'pearson':>10} {'spearman':>10} {'rank-rank':>11} {'top10 hit':>11} {'enrichment':>12}")
    for s in STUDIES:
        if s not in summary: continue
        x = summary[s]
        ench = x.get("enrichment_median")
        ench_s = f"${ench:+,.0f}" if ench is not None else "n/a"
        print(f"  {s.upper():<5} {x['pearson_train_fwd']:>+10.3f} {x['spearman_train_fwd']:>+10.3f} "
              f"{x['spearman_rank_rank']:>+11.3f} {x['top10_hit_rate']:>9d}/10 {ench_s:>12}")

    # Save
    out_path = f"{OUTDIR}/forward_distribution_analysis.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved {out_path}")

    # Interpretive summary
    print(f"\n{'='*92}")
    print(f"  INTERPRETATION")
    print(f"{'='*92}")
    print()
    print("  - HIGHER median  = objective produces consistently good trials")
    print("  - HIGHER p90     = objective has a higher ceiling among its top trials")
    print("  - HIGHER spearman_rank_rank = training score ranks trials correctly by forward")
    print("  - HIGHER top10 hit rate = picking by training score gets you forward winners")
    print("  - HIGHER enrichment = the objective actually enriches for forward performance")
    print("                         (top trials forward > random middle trials)")
    print()
    print("  If W7 wins on 'best' but W9 wins on 'spearman' + 'enrichment', then W9's objective")
    print("  produces a more reliable ranking — even if its absolute winners are smaller.")
    print("  That's a more deployable property than 'lucky one-trial superstar'.")


if __name__ == "__main__":
    main()
