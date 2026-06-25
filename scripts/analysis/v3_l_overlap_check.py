"""How many v3 R-O candidates fire INSIDE an L holding window?

Same pre-compute as Optuna v3 (defer-bar-1, color-aware), but also tracks
L windows and reports the overlap stats — both in training window and OOS.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json, os
from collections import defaultdict

STARTING_CASH = 25_000
POSITION_PCT = 0.30
BASELINE = "config/trial_432_params.json"
W21B_DEPLOY = "config/trial_w21b_511_deploy.json"
ALL_STRATS = ["h","g","a","f","d","v","p","m","r","w","o","b","k","c","s","e","i","j","n","l","x"]

WINDOWS = {
    "train_2024_to_2026feb": {"dirs": [
        "stored_data_combined", "stored_data_jan_mar_2024", "stored_data_apr_jun_2024",
        "stored_data_jul_sep_2024", "stored_data_oct_dec_2024",
        "stored_data_jan_mar_2025", "stored_data_apr_jun_2025", "stored_data_jul_2025",
        "stored_data_oos", "stored_data"],
        "lo": "2024-01-01", "hi": "2026-02-28"},
    "2022": {"dirs": ["stored_data_2022"], "lo": "2022-01-01", "hi": "2022-12-31"},
    "2026_mar_jun": {"dirs": ["stored_data_mar_may_2026", "stored_data_jun_2026"],
                    "lo": "2026-03-01", "hi": "2026-06-30"},
}


def run(tgc, dirs, lo, hi):
    from test_full import load_all_picks, MARGIN_THRESHOLD
    dirs = [d for d in dirs if os.path.exists(d)]
    all_dates, picks_by_date = load_all_picks(dirs)
    dates = sorted([d for d in all_dates if lo <= d <= hi])

    cash = STARTING_CASH
    g_holds, l_holds = defaultdict(list), defaultdict(list)
    for d in dates:
        dp = picks_by_date.get(d, [])
        if not dp: continue
        is_cash = cash < MARGIN_THRESHOLD
        try:
            states, end_c, unset, _ = tgc.simulate_day_combined(dp, cash, cash_account=is_cash)
        except Exception: continue
        for st in states:
            if st.get("exit_reason") and st.get("position_cost", 0) > 0:
                key = (st.get("ticker"), d)
                etime, xtime = st.get("entry_time"), st.get("exit_time")
                if etime is None or xtime is None: continue
                if st.get("strategy") == "G": g_holds[key].append((etime, xtime))
                elif st.get("strategy") == "L": l_holds[key].append((etime, xtime))
        cash = end_c + (unset if is_cash else 0)

    # Generate v3 candidates
    total = 0; overlap_with_l_window = 0; same_ticker_as_l = 0
    for d in dates:
        for p in picks_by_date.get(d, []):
            mh = p.get("market_hour_candles")
            if mh is None or len(mh) < 2: continue
            day_open = float(mh.iloc[0]["Open"])
            bar0_open = float(mh.iloc[0]["Open"])
            bar0_close = float(mh.iloc[0]["Close"])
            bar0_red = bar0_close <= bar0_open
            ghol = g_holds.get((p["ticker"], d), [])
            lhol = l_holds.get((p["ticker"], d), [])

            scan_start_idx = 1
            if not bar0_red and ghol:
                g_exit_ts = max(x for _, x in ghol)
                new_start = None
                for i in range(1, len(mh)):
                    if mh.index[i] > g_exit_ts: new_start = i; break
                if new_start is None: continue
                scan_start_idx = new_start

            entry_idx = None; entry_ts = None
            for i in range(scan_start_idx, len(mh)):
                row = mh.iloc[i]
                if float(row["Close"]) > day_open:
                    entry_idx = i; entry_ts = row.name; break
            if entry_idx is None: continue

            total += 1
            if lhol:
                same_ticker_as_l += 1
                if any(et <= entry_ts <= xt for et, xt in lhol):
                    overlap_with_l_window += 1
    return total, overlap_with_l_window, same_ticker_as_l


def main():
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    with open(BASELINE) as f: baseline = json.load(f)
    with open(W21B_DEPLOY) as f: p511 = json.load(f)["params"]
    merged = {**baseline, **p511}
    for s in ALL_STRATS:
        merged[f"enable_{s}"] = (s in {"g","l"})
    set_strategy_params(merged)
    tgc.USE_DYNAMIC_SLIPPAGE = True
    tgc.USE_MULTIWINDOW_SLIPPAGE = True
    tgc.USE_VOLATILITY_ADJUSTMENT = True
    tgc.SLIP_IMPACT_K = 3.0
    tgc.VOL_CAP_PCT = 5.0
    tgc.MAX_2MIN_PARTICIPATION = 0.15
    tgc.MAX_REGIME_PARTICIPATION = 0.08
    tgc.NEWS_MODULATOR_ENABLED = False

    print(f"{'window':<26} {'v3_cands':>10} {'same_tkr_L':>12} {'IN_L_window':>14} {'%inside':>8}")
    for wname, w in WINDOWS.items():
        n, ov, same = run(tgc, w["dirs"], w["lo"], w["hi"])
        pct = 100*ov/n if n else 0
        print(f"{wname:<26} {n:>10} {same:>12} {ov:>14} {pct:>7.2f}%")


if __name__ == "__main__":
    main()
