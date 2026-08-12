"""Run specific config JSONs against the 2026 OOS window.

Usage: python run_oos_two_configs.py
Runs trial #106 (2x margin) and trial #121 (1x) on 2026-03-02..2026-08-07.
"""
import json
import os
import pickle
import sys

CACHE = os.path.join("stored_data_1min", "fulltest_picks_gap2_vol250k.pkl")
STARTING_CASH = 25_000


def run_config(path, label, margin_multiplier=None, max_pos_pct=None):
    with open(CACHE, "rb") as f:
        picks_by_date = pickle.load(f)

    d = json.load(open(path, encoding="utf-8"))
    params = d.get("params", d)
    trial_no = d.get("trial_number", "?")

    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params, _build_param_snapshot

    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.USE_2MIN_SLIPPAGE = False

    set_strategy_params(params)

    # Study-time env flags (G2 studies trained with first-bar-only fills)
    if d.get("g_first_bar_only"):
        tgc.G_FIRST_BAR_ONLY = True
    if margin_multiplier is not None:
        tgc.MARGIN_MULTIPLIER = float(margin_multiplier)
    if max_pos_pct is not None:
        tgc.MAX_POSITION_PCT_OF_CASH = float(max_pos_pct)

    snapshot = _build_param_snapshot()

    dates = sorted(x for x in picks_by_date if "2026-03-02" <= x <= "2026-08-07")
    cash = float(STARTING_CASH)
    all_trades = []
    for dd in dates:
        picks = picks_by_date.get(dd, [])
        if not picks:
            continue
        cash_account = cash < tgc.MARGIN_THRESHOLD
        states, cash, unsettled, _ = tgc.simulate_day_combined(
            picks, cash, cash_account, params=snapshot
        )
        effective_cash = cash + (unsettled if cash_account else 0)
        for st in states:
            if st.get("exit_reason") is not None and st.get("position_cost", 0) > 0:
                pnl = st.get("pnl", 0)
                if pnl is not None:
                    all_trades.append(pnl)
        cash = effective_cash

    n = len(all_trades)
    if n == 0:
        print(f"{label}: 0 trades")
        return
    total_pnl = sum(all_trades)
    wins = [p for p in all_trades if p > 0]
    losses = [p for p in all_trades if p <= 0]
    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    print(f"{label} (trial {trial_no}): {n} trades, {len(wins)} wins, "
          f"WR={len(wins)/n*100:.1f}%, PnL=${total_pnl:,.0f}, PF={gross_win/gross_loss:.2f}")


if __name__ == "__main__":
    # trial #106 - tuned WITH 2x margin
    run_config(
        "config/trial_gl_1min_g2_2x_deploy.json",
        "MARGIN   #106",
        margin_multiplier=2.0,
        max_pos_pct=200.0,
    )
    # trial #121 - tuned WITHOUT margin (v4)
    run_config("config/trial_gl_1min_v4_g2_best.json", "NO-MARGIN #121")
