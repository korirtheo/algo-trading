"""Trade-by-trade live-vs-backtest audit for a single trading day.

Run AFTER market close to compare:
  - Live Alpaca fills for the day
  - Backtest output for the same day with the same config (#124)

Reports each (ticker, day) pair side-by-side:
  - Did both decide to trade it?
  - Entry timing (live fill time vs backtest signal time)
  - Notional (live vs backtest, expected ratio ~ 0.30 if equity cap binds)
  - PnL % (per-share return — should match within slippage)
  - Exit reason

Flags rows where:
  - Live traded but backtest didn't (scanner divergence)
  - Backtest traded but live didn't (missed signal)
  - Notional ratio outside [0.20, 0.40] window
  - PnL % deviation > 200bp

Usage:
    python scripts/analysis/live_vs_backtest_audit.py [YYYY-MM-DD]

If no date passed, uses TODAY in ET.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

from config import settings
import test_green_candle_combined as tgc
from optimize_combined import set_strategy_params
from test_full import load_all_picks, MARGIN_THRESHOLD

CONFIG_PATH = os.environ.get(
    "AUDIT_CONFIG_PATH",
    "config/trial_124_microcap_pump_extracted.json",
)
BASELINE = "config/trial_432_params.json"
DATA_DIRS_TODAY = ["stored_data", "stored_data_mar_may_2026"]

ET = ZoneInfo("America/New_York")


def _target_date():
    if len(sys.argv) > 1:
        return sys.argv[1]
    return datetime.now(ET).strftime("%Y-%m-%d")


def _load_live_fills(date_str):
    c = TradingClient(settings.ALPACA_API_KEY, settings.ALPACA_API_SECRET, paper=True)
    start = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=0, minute=0, tzinfo=ET).astimezone(timezone.utc)
    end = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=23, minute=59, tzinfo=ET).astimezone(timezone.utc)
    orders = c.get_orders(filter=GetOrdersRequest(
        status=QueryOrderStatus.ALL, after=start, until=end, limit=500,
    ))

    fills_by_ticker = {}
    for o in orders:
        if not o.filled_qty or float(o.filled_qty) <= 0:
            continue
        sym = o.symbol
        side = o.side.value
        qty = float(o.filled_qty)
        price = float(o.filled_avg_price or 0)
        if price <= 0:
            continue
        entry = fills_by_ticker.setdefault(sym, {
            "buys": [], "sells": [], "order_class": None})
        if side == "buy":
            entry["buys"].append((o.submitted_at, qty, price, o.id))
            entry["order_class"] = o.order_class.value if o.order_class else "simple"
        else:
            entry["sells"].append((o.submitted_at, qty, price, o.id))

    return fills_by_ticker


def _merge_baseline(params):
    with open(BASELINE) as f:
        b = json.load(f)
    m = dict(b); m.update(params); return m


def _load_backtest_trades(date_str):
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    set_strategy_params(_merge_baseline(cfg["params"]))
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.X_MIN_FIRST_LEG_GAIN_PCT = 9999.0

    dirs_present = [d for d in DATA_DIRS_TODAY if os.path.exists(d)]
    if not dirs_present:
        print(f"[error] No data dirs found ({DATA_DIRS_TODAY}). Backtest skipped.")
        return [], 0

    all_dates, picks = load_all_picks(dirs_present)
    if date_str not in all_dates:
        print(f"[warn] {date_str} not in picks data. Available range: "
              f"{all_dates[0]} -> {all_dates[-1]}")
        return [], 0

    day_picks = picks.get(date_str, [])
    if not day_picks:
        return [], 0

    # Backtest uses $25K starting cash to match a fresh paper account.
    starting_cash = 25_000
    cash = starting_cash
    is_cash = cash < MARGIN_THRESHOLD
    states, end_cash, unset, _ = tgc.simulate_day_combined(
        day_picks, cash, cash_account=is_cash)

    bt_trades = []
    for st in states:
        if st.get("exit_reason") is None or st.get("position_cost", 0) <= 0:
            continue
        bt_trades.append({
            "ticker": st["ticker"],
            "strategy": st.get("strategy"),
            "entry_time": st.get("entry_time"),
            "entry_price": st.get("entry_price"),
            "exit_price": st.get("exit_price"),
            "exit_time": st.get("exit_time"),
            "exit_reason": st.get("exit_reason"),
            "shares": st.get("shares"),
            "position_cost": st.get("position_cost"),
            "pnl": st.get("pnl"),
            "pnl_pct": (st["pnl"] / st["position_cost"] * 100) if st["position_cost"] > 0 else 0,
            "vol_capped": st.get("vol_capped", False),
        })
    return bt_trades, starting_cash


def main():
    date_str = _target_date()
    print(f"=== Live vs Backtest audit for {date_str} ===")
    print(f"Config: {CONFIG_PATH}\n")

    print("Loading live Alpaca fills...")
    live = _load_live_fills(date_str)
    print(f"  Tickers with live fills: {len(live)}")

    print("Running backtest for the same day...")
    bt_trades, starting_cash = _load_backtest_trades(date_str)
    bt_by_ticker = {t["ticker"]: t for t in bt_trades}
    print(f"  Backtest trades: {len(bt_trades)} (starting cash ${starting_cash:,})")

    # Union of tickers
    all_tickers = sorted(set(live.keys()) | set(bt_by_ticker.keys()))
    if not all_tickers:
        print("\nNo trades on either side — neither live nor backtest fired today.")
        return

    print(f"\n{'='*120}")
    print(f"  {'ticker':<8} {'side':<7} {'live_notional':>14} {'bt_notional':>14} "
          f"{'ratio':>7} {'live_pnl%':>9} {'bt_pnl%':>9} {'flag'}")
    print(f"{'='*120}")

    flag_picks_div = 0
    flag_notional_div = 0
    flag_pnl_div = 0

    for tk in all_tickers:
        lv = live.get(tk)
        bt = bt_by_ticker.get(tk)

        if lv and not bt:
            for b in lv["buys"]:
                _, qty, price, _ = b
                notional = qty * price
                print(f"  {tk:<8} {'BUY':<7} ${notional:>13,.0f} {'-':>14} {'-':>7} "
                      f"{'-':>9} {'-':>9} ONLY-LIVE (scanner divergence)")
            flag_picks_div += 1
            continue

        if bt and not lv:
            print(f"  {tk:<8} {'BUY':<7} {'-':>14} ${bt['position_cost']:>13,.0f} "
                  f"{'-':>7} {'-':>9} {bt['pnl_pct']:>8.2f}% ONLY-BACKTEST (missed signal)")
            flag_picks_div += 1
            continue

        # Both traded — compare notional and pnl
        live_notional = sum(q * p for _, q, p, _ in lv["buys"])
        bt_notional = bt["position_cost"]
        ratio = (live_notional / bt_notional) if bt_notional > 0 else 0

        # Compute live realized pnl% if we have matching sells
        live_pnl_pct_str = "n/a"
        if lv["buys"] and lv["sells"]:
            avg_buy = sum(q*p for _, q, p, _ in lv["buys"]) / sum(q for _, q, _, _ in lv["buys"])
            avg_sell = sum(q*p for _, q, p, _ in lv["sells"]) / sum(q for _, q, _, _ in lv["sells"])
            live_pnl_pct = (avg_sell / avg_buy - 1.0) * 100 if avg_buy > 0 else 0
            live_pnl_pct_str = f"{live_pnl_pct:+.2f}%"
        else:
            live_pnl_pct = None

        flags = []
        if ratio < 0.20 or ratio > 0.40:
            flags.append(f"NOTIONAL-OUT-OF-BAND")
            flag_notional_div += 1
        if live_pnl_pct is not None:
            dev_bp = abs(live_pnl_pct - bt["pnl_pct"]) * 100
            if dev_bp > 200:
                flags.append(f"PNL-DRIFT-{dev_bp:.0f}bp")
                flag_pnl_div += 1
        flag_str = " ".join(flags) if flags else "OK"
        bracket_str = "[BRACKET]" if lv["order_class"] == "bracket" else "[simple]"
        print(f"  {tk:<8} {bracket_str:<9} ${live_notional:>12,.0f} ${bt_notional:>12,.0f} "
              f"{ratio:>6.2f}x {live_pnl_pct_str:>9} {bt['pnl_pct']:>+8.2f}% {flag_str}")

    print(f"\n{'='*120}")
    print(f"Summary:")
    print(f"  Tickers both:        {len(set(live.keys()) & set(bt_by_ticker.keys()))}")
    print(f"  Only live:           {len(set(live.keys()) - set(bt_by_ticker.keys()))}")
    print(f"  Only backtest:       {len(set(bt_by_ticker.keys()) - set(live.keys()))}")
    print(f"  Scanner divergence:  {flag_picks_div}")
    print(f"  Notional out-of-band:{flag_notional_div} (expect ratio ~0.30 from equity cap)")
    print(f"  PnL drift > 200bp:   {flag_pnl_div}")

    # Bracket-order coverage
    bracket_count = sum(1 for v in live.values() if v["order_class"] == "bracket")
    print(f"  Bracket orders:      {bracket_count}/{len(live)} ({100*bracket_count/max(len(live),1):.0f}%)")
    if bracket_count < len(live):
        print(f"    [WARN] Some buys lack brackets — safety rail not fully active.")


if __name__ == "__main__":
    main()
