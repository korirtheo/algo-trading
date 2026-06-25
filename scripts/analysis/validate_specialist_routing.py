"""Test specialist-routing vs ungated-#124 across 2024, 2025, 2026.

Two routing variants tested:

  STRICT  — only trade days where we have a shape-specific specialist.
            Skip everything else.
              microcap-thin / thin-microcap -> #124
              liquid-normal                  -> #450
              (broad-squeeze, mega-cap, corp-action, dead, empty) -> SKIP

  FALLBACK — use specialist where we have one, else fall back to #124.
              microcap-thin / thin-microcap -> #124
              liquid-normal                  -> #450
              other                          -> #124 (default)

For each variant, we report total PnL AND per-regime breakdown so we can
see which regimes are pulling weight vs dragging.

Compared to the ungated #124 baseline (validate_124_multiyear.py):
    2024  $209,548  (8.38x)
    2025  $833,822  (33.35x)
    2026  $ 47,534  (1.90x)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import numpy as np

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD
from scripts.analysis.squeeze_taxonomy_2021_2026 import day_signature, classify_shape
from strategies.regime_gate import classify_regime, compute_features

MICROCAP_CONFIG = "config/trial_124_microcap_pump_extracted.json"
LIQUID_CONFIG = "config/trial_450_liquid_normal_extracted.json"
BASELINE = "config/trial_432_params.json"
STARTING_CASH = 25_000

YEAR_DIRS = {
    "2024": ["stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
              "stored_data_jul_sep_2024", "stored_data_oct_dec_2024"],
    "2025": ["stored_data_jan_mar_2025", "stored_data_apr_jun_2025",
              "stored_data_jul_2025", "stored_data_oos"],
    "2026": ["stored_data", "stored_data_mar_may_2026"],
}

ALL_DAYS_BASELINE = {
    "2024": {"final": 209_548, "multiplier": 8.38},
    "2025": {"final": 833_822, "multiplier": 33.35},
    "2026": {"final":  47_534, "multiplier":  1.90},
}

MICROCAP_SHAPES = {"microcap-thin", "thin-microcap"}
LIQUID_SHAPES = {"liquid-normal"}


def _merged(p):
    with open(BASELINE) as f: b = json.load(f)
    m = dict(b); m.update(p); return m


def _load_cfg(path):
    with open(path) as f: c = json.load(f)
    return _merged(c["params"])


def _apply_slippage_defaults():
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.X_MIN_FIRST_LEG_GAIN_PCT = 9999.0


def _classify(picks):
    try:
        feats = compute_features(picks)
        regime = classify_regime(feats)
    except Exception:
        regime = "NORMAL"
    sig = day_signature(picks)
    if sig is None:
        return "empty"
    return classify_shape(sig, regime)


def run_year(year, dirs, variant, microcap_params, liquid_params):
    """variant in {'strict', 'fallback', 'ungated'}"""
    dirs_present = [d for d in dirs if os.path.exists(d)]
    all_dates, picks = load_all_picks(dirs_present)
    test_dates = [d for d in all_dates if d.startswith(year)]

    cash = STARTING_CASH
    pnl_by_shape = {}
    n_traded_by_shape = {}
    n_skipped = 0
    current_specialist = None

    for d in test_dates:
        day_picks = picks.get(d, [])
        if not day_picks:
            continue
        shape = _classify(day_picks)

        # --- Decide which specialist to use (None = skip) ---
        if variant == "ungated":
            specialist = "micro"
        elif variant == "strict":
            if shape in MICROCAP_SHAPES:
                specialist = "micro"
            elif shape in LIQUID_SHAPES:
                specialist = "liquid"
            else:
                specialist = None
        elif variant == "fallback":
            if shape in LIQUID_SHAPES:
                specialist = "liquid"
            else:
                specialist = "micro"
        else:
            raise ValueError(variant)

        if specialist is None:
            n_skipped += 1
            continue

        # Swap params if specialist changed
        if specialist != current_specialist:
            if specialist == "micro":
                set_strategy_params(microcap_params)
            else:
                set_strategy_params(liquid_params)
            _apply_slippage_defaults()
            current_specialist = specialist

        starting = cash
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception:
            continue
        cash = end_c
        if is_cash: cash += unset

        day_pnl = cash - starting
        pnl_by_shape.setdefault(shape, 0.0)
        pnl_by_shape[shape] += day_pnl
        n_traded_by_shape[shape] = n_traded_by_shape.get(shape, 0) + 1

    return {
        "year": year,
        "variant": variant,
        "final": cash,
        "multiplier": cash / STARTING_CASH,
        "pnl_by_shape": pnl_by_shape,
        "n_traded_by_shape": n_traded_by_shape,
        "n_skipped": n_skipped,
    }


def main():
    microcap_params = _load_cfg(MICROCAP_CONFIG)
    liquid_params = _load_cfg(LIQUID_CONFIG)
    print(f"Microcap specialist: {MICROCAP_CONFIG}")
    print(f"Liquid specialist:   {LIQUID_CONFIG}\n")

    rows = []  # (year, variant, final, multiplier)
    pnl_breakdowns = []  # (year, variant, pnl_by_shape, n_traded_by_shape)

    for variant in ("ungated", "strict", "fallback"):
        print(f"=== Variant: {variant.upper()} ===")
        for year, dirs in YEAR_DIRS.items():
            r = run_year(year, dirs, variant, microcap_params, liquid_params)
            rows.append((year, variant, r['final'], r['multiplier']))
            pnl_breakdowns.append((year, variant, r['pnl_by_shape'], r['n_traded_by_shape']))
            print(f"  {year}: ${r['final']:>10,.0f} ({r['multiplier']:>5.2f}x)  "
                  f"skipped={r['n_skipped']:>3}  "
                  f"traded={sum(r['n_traded_by_shape'].values()):>3}")
        print()

    # Headline comparison
    print("=" * 88)
    print(f"  {'year':<6} {'ungated':>14} {'strict':>14} {'fallback':>14} {'best':<10}")
    print("=" * 88)
    by_year = {}
    for y, v, f, m in rows:
        by_year.setdefault(y, {})[v] = (f, m)
    for y, m in by_year.items():
        un = m['ungated'][0]; st = m['strict'][0]; fb = m['fallback'][0]
        best = max([('ungated', un), ('strict', st), ('fallback', fb)],
                   key=lambda t: t[1])[0]
        print(f"  {y:<6} ${un:>12,.0f} (vs baseline ${ALL_DAYS_BASELINE[y]['final']:,})")
        print(f"         {'strict':<14} ${st:>12,.0f}    delta vs ungated: ${st-un:+,.0f}")
        print(f"         {'fallback':<14} ${fb:>12,.0f}    delta vs ungated: ${fb-un:+,.0f}")
        print(f"         BEST: {best.upper()}\n")

    # Per-regime PnL contribution (use fallback's breakdown since it covers all regimes)
    print("=" * 88)
    print("PER-REGIME PnL CONTRIBUTION (fallback variant — #124 covers most regimes):")
    print("=" * 88)
    for y, v, pbs, ntbs in pnl_breakdowns:
        if v != "fallback": continue
        print(f"\n  {y}:")
        total = sum(pbs.values())
        for shape in sorted(pbs.keys(), key=lambda s: -pbs[s]):
            pnl = pbs[shape]
            n = ntbs.get(shape, 0)
            avg = pnl / n if n else 0
            pct = 100 * pnl / total if total else 0
            tag = "(specialist)" if shape in MICROCAP_SHAPES | LIQUID_SHAPES else ""
            print(f"    {shape:<18} {n:>3} days  "
                  f"PnL ${pnl:>+12,.0f}  avg ${avg:>+10,.0f}/day  "
                  f"({pct:>+5.1f}% of total)  {tag}")


if __name__ == "__main__":
    main()
