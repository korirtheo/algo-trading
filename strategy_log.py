"""
Strategy Version Log — tracks all optimized configs, objectives, and results.

Each version records:
  - Date optimized
  - Objective function used
  - Dataset used
  - All parameter values
  - Performance metrics (flat and compounded)
  - Notes on what changed

Run this file to print a comparison table of all versions.
"""

VERSIONS = {

    # ─────────────────────────────────────────────────────────────────────────
    "v1_phase2_manual": {
        "date": "2025-12-01",
        "description": "Original hand-tuned config — pre-Optuna baseline",
        "objective": "manual (no optimizer)",
        "dataset": "stored_data_combined (284 days, Jan 2025 - Feb 2026)",
        "regime_filter": "SPY > SMA(40)",
        "params": {
            "stop_loss_pct": 16.0,
            "partial_sell_frac": 0.90,
            "partial_sell_pct": 15.0,
            "atr_period": 8,
            "atr_multiplier": 4.25,
            "confirm_above": 2,
            "confirm_window": 4,
            "pullback_pct": 4.0,
            "pullback_timeout": 24,
            "n_exit_tranches": 3,
            "partial_sell_frac_2": 0.35,
            "partial_sell_pct_2": 25.0,
            "min_pm_volume": 250_000,
            "min_gap_pct": 2.0,
            "scale_in": 1,
            "scale_in_trigger_pct": 14.0,
            "scale_in_frac": 0.50,
            "eod_exit_minutes": 30,
            "entry_cutoff_minutes": 0,
            "runner_mode": 0,
            "liq_vacuum": 0,
            "structural_stop": 0,
            "structural_stop_atr_mult": 1.5,
        },
        "results": {
            # Aug 2025 - Feb 2026 (143 days, stored_data + stored_data_oos)
            "aug_feb_143d": {
                "starting_cash": 25_000,
                "final_equity": 227_332,
                "total_return_pct": 809,
                "win_rate_pct": 55.2,
                "total_trades": 299,
                "sharpe": None,  # not recorded
            },
            # Full year pending — backtest running
        },
        "notes": (
            "Wide stops (16%) + conservative scale-in (50% at +14%) + volume filter (250K). "
            "Compounds extremely well — $25K to $227K on 143 days. "
            "December 2025 was the explosive month: $132K -> $335K peak. "
            "Lower MIN_GAP (2%) means more trades = more diversification. "
            "This was the '$71M config' before the scale-in timing bug was fixed."
        ),
    },

    # ─────────────────────────────────────────────────────────────────────────
    "v2_phase3_optuna_total_pnl": {
        "date": "2026-03-03",
        "description": "Optuna Phase 3 — optimized for TOTAL flat P&L ($10K/day)",
        "objective": "maximize total_pnl (flat $10K/day, no compounding)",
        "optimizer": "Optuna TPE, 500 trials",
        "dataset": "stored_data_combined (284 days, Jan 2025 - Feb 2026)",
        "regime_filter": "SPY > SMA(40)",
        "params": {
            "stop_loss_pct": 11.0,
            "partial_sell_frac": 0.95,
            "partial_sell_pct": 19.0,
            "atr_period": 21,
            "atr_multiplier": 4.0,
            "confirm_above": 3,
            "confirm_window": 3,
            "pullback_pct": 5.5,
            "pullback_timeout": 20,
            "n_exit_tranches": 2,
            "partial_sell_frac_2": 0.35,
            "partial_sell_pct_2": 25.0,
            "min_pm_volume": 0,
            "min_gap_pct": 12.0,
            "scale_in": 1,
            "scale_in_trigger_pct": 5.0,
            "scale_in_frac": 1.00,
            "eod_exit_minutes": 35,
            "entry_cutoff_minutes": 0,
            "runner_mode": 0,
            "liq_vacuum": 0,
            "structural_stop": 1,
            "structural_stop_atr_mult": 1.25,
        },
        "results": {
            "full_year_284d": {
                "starting_cash": 25_000,
                "final_equity": 405_334,
                "total_return_pct": 1521,
                "win_rate_pct": 39.2,
                "total_trades": 288,
                "flat_total_pnl": None,  # was the optimizer target
                "sharpe": None,
            },
            "aug_feb_143d": {
                "starting_cash": 25_000,
                "final_equity": 64_913,
                "total_return_pct": 160,
                "win_rate_pct": 39.2,
                "total_trades": 288,
                "sharpe": None,
            },
        },
        "notes": (
            "Optuna maximized flat total P&L on $10K/day — no compounding in objective. "
            "Aggressive scale-in (100% at +5%) with tight stops (11%) maximizes flat P&L "
            "but compounds poorly: big concentrated losses compound negatively. "
            "December 2025: only +$14K vs old config's +$149K. "
            "Higher MIN_GAP (12%) = fewer trades = less diversification. "
            "Structural stop adds extra kill layer that hurts with tight stops."
        ),
    },

    # ─────────────────────────────────────────────────────────────────────────
    # v3 will be filled after expectancy optimization
    "v3_optuna_expectancy": {
        "date": "2026-03-05",
        "description": "Optuna Phase 4 — optimized for EXPECTANCY per trade",
        "objective": "maximize expectancy = total_pnl / num_trades",
        "optimizer": "Optuna TPE, 500 trials",
        "dataset": "stored_data_combined (284 days, Jan 2025 - Feb 2026)",
        "regime_filter": "SPY > SMA(40)",
        "params": {},  # TO BE FILLED after optimization
        "results": {},  # TO BE FILLED after optimization
        "notes": "Pending — optimization not yet run.",
    },
}

# ─── Regime filter optimization results ──────────────────────────────────────
REGIME_RESULTS = {
    "v1_spy_sma40": {
        "date": "2026-03-04",
        "ticker": "SPY",
        "sma_period": 40,
        "vix_threshold": None,
        "result_pnl": 30_418,
        "notes": "Original default. Decent but SPY doesn't track small-cap momentum well.",
    },
    "v2_iwm_sma35_vix25": {
        "date": "2026-03-05",
        "ticker": "IWM",
        "sma_period": 35,
        "vix_threshold": 25,
        "result_pnl": 32_029,
        "sharpe": 3.17,
        "notes": (
            "Optimized with 200 Optuna trials. IWM (small-cap index) better matches "
            "our small-cap gap-up universe. Top 10 trials all converged on IWM SMA(35). "
            "Not yet applied to test_full.py."
        ),
    },
}


# ─── Print comparison ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"{'='*90}")
    print(f"  STRATEGY VERSION LOG — {len(VERSIONS)} versions")
    print(f"{'='*90}\n")

    for name, v in VERSIONS.items():
        print(f"  [{name}]")
        print(f"  {v['description']}")
        print(f"  Date: {v['date']}")
        print(f"  Objective: {v['objective']}")
        print(f"  Dataset: {v['dataset']}")
        print(f"  Regime: {v['regime_filter']}")
        print()

        # Key params
        p = v["params"]
        if p:
            print(f"    {'Param':<28} {'Value':>10}")
            print(f"    {'-'*40}")
            for k, val in p.items():
                print(f"    {k:<28} {str(val):>10}")
        else:
            print(f"    (params pending)")

        # Results
        print()
        for rname, r in v.get("results", {}).items():
            if r:
                eq = r.get("final_equity")
                ret = r.get("total_return_pct")
                wr = r.get("win_rate_pct")
                trades = r.get("total_trades")
                print(f"    [{rname}] ${eq:,} ({ret:+,}%) | WR {wr}% | {trades} trades" if eq else f"    [{rname}] pending")
            else:
                print(f"    [{rname}] pending")

        print(f"\n    Notes: {v['notes']}\n")
        print(f"  {'-'*86}\n")

    # Param diff table
    v1 = VERSIONS["v1_phase2_manual"]["params"]
    v2 = VERSIONS["v2_phase3_optuna_total_pnl"]["params"]
    v3 = VERSIONS["v3_optuna_expectancy"]["params"]

    print(f"\n{'='*90}")
    print(f"  PARAMETER COMPARISON")
    print(f"{'='*90}")
    print(f"  {'Param':<28} {'v1 (manual)':>12} {'v2 (tot_pnl)':>14} {'v3 (expect)':>14}")
    print(f"  {'-'*70}")

    all_keys = list(v1.keys())
    for k in all_keys:
        val1 = v1.get(k, "—")
        val2 = v2.get(k, "—")
        val3 = v3.get(k, "—") if v3 else "—"
        marker = " <--" if val1 != val2 else ""
        print(f"  {k:<28} {str(val1):>12} {str(val2):>14} {str(val3):>14}{marker}")

    # Regime results
    print(f"\n{'='*90}")
    print(f"  REGIME FILTER RESULTS")
    print(f"{'='*90}")
    for name, r in REGIME_RESULTS.items():
        print(f"  [{name}] {r['ticker']} SMA({r['sma_period']}) "
              f"VIX<{r.get('vix_threshold', 'N/A')} "
              f"-> ${r['result_pnl']:,} P&L | {r['notes'][:60]}...")
