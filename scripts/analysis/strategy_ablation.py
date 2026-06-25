"""Strategy ablation — measure forward contribution of each enabled strategy
within the W7 #254 winner (and a couple of comparison configs).

Per ChatGPT's diagnostic:
  Test #254 - L  -> if forward drops dramatically, L is the ceiling
  Test #254 - R  -> if forward drops dramatically, R is the ceiling
  Test #254 - L - R -> the genuine non-squeeze contribution
  Plus single-strategy isolations
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
DATA_DIRS_2026 = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]
OUTDIR = "results"


# Base configs to ablate
CONFIGS = [
    {"label": "#254 W7",   "path": "config/trial_254_w7_extracted.json"},
    {"label": "#124 W3",   "path": "config/trial_124_microcap_pump_extracted.json"},
]

ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]


def enabled_strats(params):
    return [s for s in ALL_STRATS if params.get(f"enable_{s}") is True]


def make_ablation_params(base_params, disable=(), enable_only=None):
    """Return a copy of base_params with specified strategies disabled.
    If enable_only is set, only those strategies are enabled."""
    p = dict(base_params)
    if enable_only is not None:
        for s in ALL_STRATS:
            p[f"enable_{s}"] = (s in enable_only)
    else:
        for s in disable:
            p[f"enable_{s}"] = False
    return p


def forward(task):
    """Forward-test a single ablation config on 2026.
    task = {"label": str, "params": dict}
    """
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD

    with open(BASELINE) as f: baseline = json.load(f)
    merged = dict(baseline); merged.update(task["params"])
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
    n_trades = 0
    by_strat = {}
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
                strat = st.get("strategy", "?")
                if strat not in by_strat:
                    by_strat[strat] = {"n": 0, "pnl": 0.0}
                by_strat[strat]["n"] += 1
                by_strat[strat]["pnl"] += st["pnl"]
        cash = end_c + (unset if is_cash else 0)
        daily_eq.append(cash)
    eq = np.array(daily_eq)
    peak = np.maximum.accumulate(eq)
    dd_pct = float((eq - peak).min() / peak[(eq - peak).argmin()] * 100) if len(peak) and peak[(eq - peak).argmin()] > 0 else 0
    return {
        "label": task["label"],
        "forward_pnl": float(eq[-1] - STARTING_CASH),
        "forward_multi": float(eq[-1] / STARTING_CASH),
        "forward_dd_pct": dd_pct,
        "n_trades": n_trades,
        "per_strategy": by_strat,
    }


def build_tasks(label, params):
    """Build ablation tasks for one config."""
    enabled = enabled_strats(params)
    tasks = []
    # Baseline (no ablation)
    tasks.append({"label": f"{label} BASE ({','.join(s.upper() for s in enabled)})",
                  "params": dict(params)})
    # Remove each strategy individually
    for s in enabled:
        new_params = make_ablation_params(params, disable=[s])
        new_enabled = [x for x in enabled if x != s]
        tasks.append({"label": f"{label} -{s.upper()}  ({','.join(x.upper() for x in new_enabled)})",
                      "params": new_params})
    # Common combo ablations for #254
    if "254" in label:
        # Remove both L and R
        if "l" in enabled and "r" in enabled:
            new_params = make_ablation_params(params, disable=["l", "r"])
            new_enabled = [x for x in enabled if x not in ["l", "r"]]
            tasks.append({"label": f"{label} -L-R  ({','.join(x.upper() for x in new_enabled)})",
                          "params": new_params})
        # Only L (isolate L's contribution)
        if "l" in enabled:
            new_params = make_ablation_params(params, enable_only=["l"])
            tasks.append({"label": f"{label} ONLY L",
                          "params": new_params})
        # Only R
        if "r" in enabled:
            new_params = make_ablation_params(params, enable_only=["r"])
            tasks.append({"label": f"{label} ONLY R",
                          "params": new_params})
        # Only G (the workhorse)
        if "g" in enabled:
            new_params = make_ablation_params(params, enable_only=["g"])
            tasks.append({"label": f"{label} ONLY G",
                          "params": new_params})
        # L + R only
        if "l" in enabled and "r" in enabled:
            new_params = make_ablation_params(params, enable_only=["l", "r"])
            tasks.append({"label": f"{label} ONLY L+R",
                          "params": new_params})
    return tasks


def main():
    print(f"\n{'='*92}")
    print(f"  STRATEGY ABLATION — what is each enabled strategy contributing?")
    print(f"{'='*92}")

    all_tasks = []
    for cfg in CONFIGS:
        if not os.path.exists(cfg["path"]):
            print(f"  [skip] {cfg['path']} not found"); continue
        with open(cfg["path"]) as f: data = json.load(f)
        params = data.get("params", data)
        en = enabled_strats(params)
        print(f"\n  {cfg['label']}: enables {','.join(s.upper() for s in en)} ({len(en)} strategies)")
        all_tasks.extend(build_tasks(cfg["label"], params))

    print(f"\n  Total ablation tests: {len(all_tasks)}")
    print(f"  Forward-testing in parallel (6 workers)...\n")

    results = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(forward, t): t["label"] for t in all_tasks}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                results.append(r)
                print(f"  {r['label']:<55}  fwd ${r['forward_pnl']:>+9,.0f}  ({r['forward_multi']:.2f}x)  DD {r['forward_dd_pct']:>5.1f}%  trades {r['n_trades']:>3}")
            except Exception as e:
                print(f"  failed: {e}")

    out_path = f"{OUTDIR}/strategy_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved {out_path}")

    # Group + summarize #254 results
    p254 = [r for r in results if "#254" in r["label"]]
    p254.sort(key=lambda r: -r["forward_pnl"])

    print(f"\n\n{'='*92}")
    print(f"  #254 W7 RANKED BY FORWARD PnL")
    print(f"{'='*92}")
    print(f"  {'config':<55} {'fwd PnL':>12} {'multi':>7} {'DD%':>7} {'trades':>7}")
    for r in p254:
        print(f"  {r['label']:<55} ${r['forward_pnl']:>+10,.0f} {r['forward_multi']:>6.2f}x {r['forward_dd_pct']:>6.1f}% {r['n_trades']:>6}")

    # Per-strategy attribution from #254 BASE (the original)
    base = next((r for r in p254 if "BASE" in r["label"]), None)
    if base and base.get("per_strategy"):
        print(f"\n  #254 BASE — per-strategy contribution to +${base['forward_pnl']:,.0f}:")
        sps = sorted(base["per_strategy"].items(), key=lambda x: -x[1]["pnl"])
        print(f"  {'strat':<6} {'n':>5} {'pnl$':>11} {'share':>8} {'avg/trade$':>12}")
        total = sum(v["pnl"] for v in base["per_strategy"].values())
        for s, v in sps:
            share = v["pnl"] / total * 100 if total else 0
            avg = v["pnl"] / v["n"] if v["n"] else 0
            print(f"  {s:<6} {v['n']:>5} ${v['pnl']:>+9,.0f} {share:>+6.1f}% ${avg:>+10,.0f}")


if __name__ == "__main__":
    main()
