"""Backtest the CURRENT best trial from the running PF microcap-pump Optuna study.

Runs on 2024 picks (the held-out test year for the W3 walk-forward window).
Uses the SAME slippage stack the Optuna training used:
  - USE_DYNAMIC_SLIPPAGE = True
  - USE_MULTIWINDOW_SLIPPAGE = True
  - USE_VOLATILITY_ADJUSTMENT = True
  - SLIP_IMPACT_K = 3.0
  - VOL_CAP_PCT = 5.0
  - MAX_2MIN_PARTICIPATION = 0.15
  - MAX_REGIME_PARTICIPATION = 0.08
  - X DISABLED (X_MIN_FIRST_LEG_GAIN_PCT = 9999)

Note: pulls best from LIVE DB, so the current best may change as Optuna runs.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

DB = "results/wf_pf_microcap_pump_noX/W3_train_2021_2022_2023_test_2024.db"
BASELINE = "config/trial_432_params.json"
TEST_DIRS = ["stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
             "stored_data_jul_sep_2024", "stored_data_oct_dec_2024"]
STARTING_CASH = 25_000
OUT_DIR = "results/wf_pf_microcap_pump_noX/charts_current_best"


def _merged_params(best_params):
    with open(BASELINE) as f:
        b = json.load(f)
    m = dict(b)
    m.update(best_params)
    return m


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Pull current best from the running study
    s = optuna.load_study(
        study_name=optuna.get_all_study_names(f"sqlite:///{DB}")[0],
        storage=f"sqlite:///{DB}",
    )
    b = s.best_trial
    a = b.user_attrs
    print(f"CURRENT BEST: trial #{b.number}")
    print(f"  Score:   ${b.value:,.0f}")
    print(f"  PnL:     ${a.get('total_pnl',0):+,.0f}")
    print(f"  PF:      {a.get('pf','-')}")
    print(f"  Sharpe%: {a.get('sharpe_pct','-')}")
    print(f"  WR:      {a.get('wr','-')}%")
    print(f"  Trades:  {a.get('n','-')}")
    print(f"  Enabled: {a.get('enabled','-')}")

    # Apply the same flags the training used
    set_strategy_params(_merged_params(dict(b.params)))
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.X_MIN_FIRST_LEG_GAIN_PCT = 9999.0  # keep X disabled to match training

    print(f"\nLoading 2024 picks...")
    dirs = [d for d in TEST_DIRS if os.path.exists(d)]
    all_dates, picks = load_all_picks(dirs)
    test_dates = [d for d in all_dates if d.startswith("2024")]
    print(f"  {len(test_dates)} test days  {test_dates[0]} -> {test_dates[-1]}")

    cash = STARTING_CASH
    eq_curve = [cash]
    daily = []
    all_trades = []
    print(f"\nSimulating...")
    for d in test_dates:
        day_picks = picks.get(d, [])
        if not day_picks:
            eq_curve.append(cash)
            daily.append({"date": d, "pnl": 0, "equity": cash, "n_trades_today": 0})
            continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(
                day_picks, cash, cash_account=is_cash)
        except Exception as e:
            eq_curve.append(cash)
            daily.append({"date": d, "pnl": 0, "equity": cash, "n_trades_today": 0,
                          "error": str(e)[:60]})
            continue
        n_today = 0
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                n_today += 1
                all_trades.append({
                    "date": d,
                    "ticker": st["ticker"],
                    "strategy": st.get("strategy"),
                    "pnl": st["pnl"],
                    "exit_reason": st.get("exit_reason"),
                    "position_cost": st["position_cost"],
                    "pct": (st["pnl"] / st["position_cost"] * 100)
                           if st["position_cost"] > 0 else 0,
                })
        pnl = end_c - cash
        cash = end_c
        if is_cash:
            cash += unset
        eq_curve.append(cash)
        daily.append({"date": d, "pnl": pnl, "equity": cash, "n_trades_today": n_today})

    eq_curve = np.array(eq_curve)
    daily_pnls = np.array([r["pnl"] for r in daily])
    final = eq_curve[-1]
    total_pnl = final - STARTING_CASH

    # Daily % returns for honest Sharpe
    prev = eq_curve[:-1]; curr = eq_curve[1:]
    mask = prev > 0
    ret = np.where(mask, (curr - prev) / prev * 100, 0)
    sharpe_pct = ret.mean()/ret.std()*np.sqrt(252) if ret.std() > 0 else 0.0
    wins_pct = (ret > 0).sum()
    losses_pct = (ret < 0).sum()

    print(f"\n{'='*72}")
    print(f"FORWARD 2024 BACKTEST RESULT (PF microcap best #{b.number})")
    print(f"{'='*72}")
    print(f"  Start equity:  ${STARTING_CASH:,.0f}")
    print(f"  Final equity:  ${final:,.0f}  ({final/STARTING_CASH:.2f}× return)")
    print(f"  Total PnL:     ${total_pnl:+,.0f}")
    print(f"  Sharpe%:       {sharpe_pct:.2f}  (on daily % returns)")
    print(f"  Win days:      {wins_pct}/{len(ret)} ({100*wins_pct/len(ret):.1f}%)")
    print(f"  Loss days:     {losses_pct}/{len(ret)} ({100*losses_pct/len(ret):.1f}%)")
    print(f"  Total trades:  {len(all_trades)}")

    # Per-strategy breakdown
    print(f"\nPer-strategy breakdown:")
    by_strat = {}
    for t in all_trades:
        s = t["strategy"]
        if s not in by_strat:
            by_strat[s] = {"n": 0, "wins": 0, "pnl": 0.0}
        by_strat[s]["n"] += 1
        by_strat[s]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            by_strat[s]["wins"] += 1
    for s, v in sorted(by_strat.items(), key=lambda x: -x[1]["pnl"]):
        wr = v["wins"] / v["n"] * 100 if v["n"] > 0 else 0
        print(f"  {s:<3}  trades={v['n']:>3}  wins={v['wins']:>3} ({wr:>5.1f}%)  pnl=${v['pnl']:>+11,.0f}")

    # Concentration (top 3 days in % returns)
    ret_sorted = np.sort(ret)[::-1]
    top3_share = ret_sorted[:3].sum() / max(0.001, ret.sum()) * 100 if ret.sum() > 0 else 0
    top10_share = ret_sorted[:10].sum() / max(0.001, ret.sum()) * 100 if ret.sum() > 0 else 0
    print(f"\nReturn concentration (% returns):")
    print(f"  Top 3 days:  {top3_share:.1f}% of total return")
    print(f"  Top 10 days: {top10_share:.1f}% of total return")
    print(f"  Median day:  {np.median(ret):+.2f}%")
    print(f"  Mean day:    {ret.mean():+.2f}%")

    # ---- CHARTS ----
    # 1. Equity curve (log scale)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(range(len(eq_curve)), eq_curve, color="#d62728", linewidth=1.8)
    ax.axhline(STARTING_CASH, color="#aaa", linestyle="--", linewidth=0.8,
               label=f"start ${STARTING_CASH:,}")
    ax.set_yscale("log")
    ax.set_title(f"PF microcap-pump best #{b.number} — forward 2024 equity (log)")
    ax.set_xlabel("Trading day index")
    ax.set_ylabel("Equity ($, log scale)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "equity_log.png")
    fig.savefig(p1, dpi=140); plt.close(fig)
    print(f"\nWrote {p1}")

    # 2. Equity curve (linear scale)
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(range(len(eq_curve)), eq_curve, color="#d62728", linewidth=1.8)
    ax.axhline(STARTING_CASH, color="#aaa", linestyle="--", linewidth=0.8,
               label=f"start ${STARTING_CASH:,}")
    ax.set_title(f"PF microcap-pump best #{b.number} — forward 2024 equity (linear)")
    ax.set_xlabel("Trading day index")
    ax.set_ylabel("Equity ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "equity_linear.png")
    fig.savefig(p2, dpi=140); plt.close(fig)
    print(f"Wrote {p2}")

    # 3. Per-day dollar PnL bars (green=win, red=loss)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#2ca02c" if p > 0 else ("#d62728" if p < 0 else "#aaa") for p in daily_pnls]
    ax.bar(range(len(daily_pnls)), daily_pnls, color=colors, alpha=0.85)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_title(f"PF microcap-pump best #{b.number} — daily PnL ($)")
    ax.set_xlabel("Trading day index")
    ax.set_ylabel("Daily PnL ($)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    fig.tight_layout()
    p3 = os.path.join(OUT_DIR, "perday_pnl.png")
    fig.savefig(p3, dpi=140); plt.close(fig)
    print(f"Wrote {p3}")

    # 4. Per-day % return bars (the honest measure)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors2 = ["#2ca02c" if r > 0 else ("#d62728" if r < 0 else "#aaa") for r in ret]
    ax.bar(range(len(ret)), ret, color=colors2, alpha=0.85)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_title(f"PF microcap-pump best #{b.number} — daily % return")
    ax.set_xlabel("Trading day index")
    ax.set_ylabel("Daily % return")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p4 = os.path.join(OUT_DIR, "perday_pct.png")
    fig.savefig(p4, dpi=140); plt.close(fig)
    print(f"Wrote {p4}")

    # 5. Distribution of % returns
    fig, ax = plt.subplots(figsize=(11, 5))
    nz = ret[ret != 0]
    if len(nz):
        ax.hist(nz, bins=60, color="#1f77b4", alpha=0.8, edgecolor="#fff")
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.axvline(np.median(ret), color="#2ca02c", linestyle="--", linewidth=1.2,
               label=f"median {np.median(ret):+.2f}%")
    ax.axvline(ret.mean(), color="#ff7f0e", linestyle="--", linewidth=1.2,
               label=f"mean {ret.mean():+.2f}%")
    ax.set_title(f"PF microcap-pump best #{b.number} — daily % return distribution")
    ax.set_xlabel("Daily % return")
    ax.set_ylabel("Day count")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p5 = os.path.join(OUT_DIR, "return_distribution.png")
    fig.savefig(p5, dpi=140); plt.close(fig)
    print(f"Wrote {p5}")

    # Save the params used
    out_meta = os.path.join(OUT_DIR, f"trial_{b.number}_meta.json")
    with open(out_meta, "w") as f:
        json.dump({
            "trial_number": b.number,
            "score": b.value,
            "user_attrs": dict(a),
            "params": dict(b.params),
            "forward_summary": {
                "final_equity": float(final),
                "total_pnl": float(total_pnl),
                "sharpe_pct": float(sharpe_pct),
                "n_trades": len(all_trades),
                "win_days": int(wins_pct),
                "loss_days": int(losses_pct),
                "top3_share_pct": float(top3_share),
                "top10_share_pct": float(top10_share),
            },
        }, f, indent=2)
    print(f"Wrote {out_meta}")


if __name__ == "__main__":
    main()
