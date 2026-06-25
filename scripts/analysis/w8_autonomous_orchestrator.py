"""Autonomous multi-round orchestrator: W8 -> W9 -> W10 if needed.

Each round runs the walk-forward Optuna with a different CV-objective variant
to see if any produces a trial that forward-tests well on 2026.

Strategy:
  W8 = default CV: score = min(year_pnl * min(year_pf, 3))
  W9 = variant chosen based on W8 failure mode
  W10 = the other variant (if W9 also disappoints)

Stopping conditions:
  * Round winner forward-tests >= $150K -> STOP, report success
  * Max rounds (3) reached -> STOP, report best of all
  * All 3 variants tried -> STOP

Decision tree per round:
  WIN          winner_fwd >= $150K           -> stop, success
  BORING       winner_fwd < $80K AND
               median(year_n) < 30           -> next round = activity_weighted
  OVERFIT      winner_fwd < $80K AND
               min_pf > 0.7                  -> next round = hybrid_sum_minpf2
  INCONCLUSIVE                               -> stop (no clear remedy)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import json
import os
import time
import subprocess
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


# -------------- config --------------
ROUNDS = [
    {"name": "W8",  "db": "optuna_w8",  "variant": None,
     "outdir": "results/walk_forward_v8_cv",  "log_subdir": "logs/w8_cv",
     "already_running": True},   # W8 workers were running (now done)
    {"name": "W9",  "db": "optuna_w9",  "variant": "geomean_year_score",
     "outdir": "results/walk_forward_v9_cv",  "log_subdir": "logs/w9_cv",
     "already_running": True},   # W9 workers already launched 2026-06-19
    # W10 deliberately removed 2026-06-19 — user wants to review W9 results
    # before deciding whether to launch W10 with hybrid_sum_minpf2 or stop.
]
TARGET_TRIALS = 600
POLL_INTERVAL = 300              # seconds (5 min)
MAX_WAIT_HOURS_PER_ROUND = 8
BASELINE_254_FORWARD = 199_553
SUCCESS_THRESHOLD = 150_000      # forward PnL that ends iteration
GOAL_FLOOR = 80_000              # forward PnL below this triggers redo

STARTING_CASH = 25_000
BASELINE = "config/trial_432_params.json"
DATA_DIRS_2026 = ["stored_data", "stored_data_mar_may_2026", "stored_data_jun_2026"]

LOG_FILE = "results/walk_forward_v8_cv/orchestrator.log"
FINAL_REPORT = "results/walk_forward_v8_cv/final_report.md"


# -------------- utilities --------------
def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def query_state(db):
    """Get trial state counts + top trials from a Postgres optuna DB."""
    try:
        c = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres",
                              dbname=db, connect_timeout=10)
    except Exception as e:
        return {"complete": 0, "running": 0, "pruned": 0, "top": [], "err": str(e)[:80]}
    cur = c.cursor()
    try:
        cur.execute("SELECT state, count(*) FROM trials GROUP BY state")
        states = dict(cur.fetchall())
    except Exception:
        c.close()
        return {"complete": 0, "running": 0, "pruned": 0, "top": []}
    done = states.get("COMPLETE", 0)
    running = states.get("RUNNING", 0)
    pruned = states.get("PRUNED", 0)
    cur.execute("""
        SELECT t.trial_id, t.number, tv.value
          FROM trials t JOIN trial_values tv ON tv.trial_id=t.trial_id
         WHERE t.state='COMPLETE' AND tv.value > -1e10
         ORDER BY tv.value DESC LIMIT 10
    """)
    top_rows = cur.fetchall()
    top = []
    for tid, num, score in top_rows:
        cur.execute("SELECT key, value_json FROM trial_user_attributes WHERE trial_id=%s", (tid,))
        ua = dict(cur.fetchall())
        top.append({"trial_id": tid, "number": num, "score": float(score),
                    "user_attrs": ua})
    c.close()
    return {"complete": done, "running": running, "pruned": pruned, "top": top}


def fetch_top_with_params(db, n):
    """Pull top N trials + full params from Postgres."""
    c = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres",
                          dbname=db, connect_timeout=10)
    cur = c.cursor()
    cur.execute("""
        SELECT t.trial_id, t.number, tv.value
          FROM trials t JOIN trial_values tv ON tv.trial_id=t.trial_id
         WHERE t.state='COMPLETE' AND tv.value > -1e10
         ORDER BY tv.value DESC LIMIT %s
    """, (n,))
    rows = cur.fetchall()
    out = []
    for tid, num, score in rows:
        cur.execute("SELECT key, value_json FROM trial_user_attributes WHERE trial_id=%s", (tid,))
        ua = dict(cur.fetchall())
        cur.execute("SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id=%s", (tid,))
        params = {}
        for name, val, dist in cur.fetchall():
            try:
                d = json.loads(dist) if dist else {}
                kind = d.get("name", "")
                if kind == "CategoricalDistribution":
                    params[name] = d["attributes"]["choices"][int(val)]
                elif "Int" in kind:
                    params[name] = int(val)
                else:
                    params[name] = float(val)
            except Exception:
                params[name] = val
        out.append({"trial_id": tid, "number": num, "score": float(score),
                    "user_attrs": ua, "params": params})
    c.close()
    return out


def create_db_if_missing(db):
    c = psycopg2.connect(host="127.0.0.1", port=5432, user="postgres", dbname="postgres")
    c.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = c.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (db,))
    if not cur.fetchone():
        cur.execute(f"CREATE DATABASE {db}")
    c.close()


def launch_workers(db, variant, outdir, log_subdir, n_workers=6):
    """Launch n_workers walk_forward_optuna processes directly (no shell).

    Avoids MSYS/bash path mangling on Windows by skipping the shell entirely.
    Each worker is a direct python.exe subprocess; cwd + env inherited from
    orchestrator.
    """
    import sys
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(log_subdir, exist_ok=True)

    python_exe = sys.executable  # full Windows path to running python

    args_base = [
        python_exe, "-u", "scripts/analysis/walk_forward_optuna.py",
        "--only-window", "5", "--n-trials", str(TARGET_TRIALS), "--n-startup", "50",
        "--shape-filter", "microcap-thin,thin-microcap", "--n-jobs", "1",
        "--use-multiwindow-slippage", "--cv-objective",
        "--storage-url", f"postgresql://postgres@127.0.0.1:5432/{db}",
        "--outdir", outdir,
    ]

    base_env = os.environ.copy()
    base_env["PHASE_NEWS_ENABLED"] = "1"
    if variant:
        base_env["W9_VARIANT"] = variant

    for i in range(1, n_workers + 1):
        log_path = f"{log_subdir}/worker_{i}.log"
        logf = open(log_path, "w")
        subprocess.Popen(
            args_base,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=base_env,
            cwd=os.getcwd(),
        )
        time.sleep(2.5)


def forward_2026_one(trial):
    """Forward-test on 2026 — designed for ProcessPoolExecutor."""
    import test_green_candle_combined as tgc
    from optimize_combined import set_strategy_params
    from test_full import load_all_picks, MARGIN_THRESHOLD

    with open(BASELINE) as f: baseline = json.load(f)
    merged = dict(baseline); merged.update(trial["params"])
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
        cash = end_c + (unset if is_cash else 0)
        daily_eq.append(cash)
    eq = np.array(daily_eq)
    peak = np.maximum.accumulate(eq)
    dd_pct = float((eq - peak).min() / peak[(eq - peak).argmin()] * 100) if len(peak) and peak[(eq - peak).argmin()] > 0 else 0
    return {
        "trial_number": trial["number"],
        "score": float(trial["score"]),
        "user_attrs": trial["user_attrs"],
        "forward_final": float(eq[-1]),
        "forward_pnl": float(eq[-1] - STARTING_CASH),
        "forward_multi": float(eq[-1] / STARTING_CASH),
        "forward_max_dd_pct": dd_pct,
        "forward_n_trades": n_trades,
    }


def diagnose(winner_forward):
    """Decide failure mode + recommended next variant.

    Escalation chain when default CV fails:
      1. geomean_year_score   (preserves CV; rewards strong years)
      2. hybrid_sum_minpf2    (last resort; re-introduces sum_pnl signal)
      3. activity_weighted    (only when trade counts collapse, rare)

    USER OVERRIDE 2026-06-19: ALWAYS continue to W9 even on WIN/INCONCLUSIVE.
    User wants the geomean variant tested regardless of W8 result.
    """
    fwd = winner_forward["forward_pnl"]
    ua = winner_forward["user_attrs"]

    year_ns = []
    for y in ["2022", "2023", "2024", "2025"]:
        v = ua.get(f"n_{y}")
        if v is not None:
            try: year_ns.append(int(v))
            except: pass
    median_n = float(np.median(year_ns)) if year_ns else 0

    try: min_pf = float(ua.get("min_pf", 0))
    except: min_pf = 0

    if fwd >= SUCCESS_THRESHOLD:
        # User-requested override: continue to W9 even on WIN.
        # Record verdict as WIN but signal next variant so loop continues.
        return "WIN_BUT_CONTINUE", "geomean_year_score"
    if fwd < GOAL_FLOOR and median_n < 30:
        return "BORING_LOW_ACTIVITY", "activity_weighted"
    if fwd < GOAL_FLOOR and min_pf > 0.7:
        return "OVERFIT_GOODHART", "geomean_year_score"
    # Used to be INCONCLUSIVE (no remedy). User wants W9 anyway.
    return "INCONCLUSIVE_FORCE_NEXT", "geomean_year_score"


def wait_for_round(db, target, max_hours):
    """Poll until COMPLETE >= target or timeout."""
    deadline = datetime.now() + timedelta(hours=max_hours)
    last_done = -1
    while datetime.now() < deadline:
        s = query_state(db)
        done = s["complete"]
        if done != last_done:
            last_done = done
            top_str = ""
            if s["top"]:
                t0 = s["top"][0]
                top_str = f"  top #{t0['number']} score={t0['score']:,.0f}"
            log(f"  {db}: {done}/{target}  running={s['running']}  pruned={s['pruned']}{top_str}")
        if done >= target:
            log(f"  {db} reached target {target}")
            return s
        time.sleep(POLL_INTERVAL)
    log(f"  {db} TIMED OUT after {max_hours}h at {last_done} trials")
    return query_state(db)


def run_round(name, db, variant, outdir, log_subdir, already_running):
    """Full round: launch (if needed) -> wait -> forward-test top 5 -> diagnose."""
    log(f"\n=== ROUND {name} (db={db}, variant={variant or 'default'}) ===")
    if not already_running:
        create_db_if_missing(db)
        log(f"  Launching 6 workers for {name}...")
        launch_workers(db, variant, outdir, log_subdir, n_workers=6)
        log(f"  Workers spawned, waiting 60s for them to load picks...")
        time.sleep(60)

    s = wait_for_round(db, TARGET_TRIALS, MAX_WAIT_HOURS_PER_ROUND)
    # Give last RUNNING trials time to settle
    time.sleep(60)
    s = query_state(db)

    # Forward-test top 5
    log(f"  Fetching top 5 with params...")
    trials = fetch_top_with_params(db, n=5)
    log(f"  Forward-testing top {len(trials)} on 2026 (4 workers)...")
    forwards = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(forward_2026_one, t): t["number"] for t in trials}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                forwards.append(r)
                log(f"  #{r['trial_number']} fwd: ${r['forward_pnl']:+,.0f} ({r['forward_multi']:.2f}x) DD {r['forward_max_dd_pct']:.1f}%")
            except Exception as e:
                log(f"  forward failed: {e}")
    forwards.sort(key=lambda r: -r["forward_pnl"])

    if not forwards:
        log(f"  No forwards completed for {name}, treating as failed round")
        return {"name": name, "db": db, "variant": variant, "state": s,
                "forwards": [], "verdict": ("INCONCLUSIVE", None)}

    winner = forwards[0]
    verdict, next_variant = diagnose(winner)
    log(f"  {name} winner: #{winner['trial_number']}  fwd ${winner['forward_pnl']:+,.0f}  verdict={verdict}")
    return {"name": name, "db": db, "variant": variant, "state": s,
            "forwards": forwards, "verdict": (verdict, next_variant)}


def _verdict_explanation(verdict, variant):
    explanations = {
        "WIN": "Round winner forward-tested >= $150K. No further iterations needed.",
        "BORING_LOW_ACTIVITY": ("Round winner forward < $80K AND median trades/year < 30. "
                    "The min(year_pnl×min(year_pf,3)) objective found a low-activity "
                    "strategy that hits the per-year floor narrowly. "
                    "Fix: add activity multiplier so TPE rewards trading more, not less."),
        "OVERFIT_GOODHART": ("Round winner forward < $80K AND min_pf > 0.7. "
                     "Classic Goodhart: TPE found the cheapest way to satisfy the min-year "
                     "constraint (trade frequently with tiny per-trade edge) — every year clears "
                     "the threshold but no year has real alpha. "
                     "Fix: switch from min() to geomean(year_pnl × min(year_pf, 3)). "
                     "Geomean still punishes weak years on log-scale but REWARDS strong years, "
                     "so high-edge configs can't be dominated by 'barely positive everywhere' trash."),
        "INCONCLUSIVE": ("Round winner forward $80K-$150K. Better than legacy objective "
                          "but not a decisive win. No clear remedy from this signal alone."),
    }
    return explanations.get(verdict, "")


def _objective_explanation(variant):
    explanations = {
        None: ("score = min(year_pnl × min(year_pf, 3.0)) across [2022,2023,2024,2025]. "
                "Each year backtested independently with fresh $25K. "
                "Punishes the WORST training year, forcing TPE to optimize regime robustness. "
                "Per-year min activity floor: 20 trades/year."),
        "default": ("score = min(year_pnl × min(year_pf, 3.0)) across all years. "
                     "Per-year reset. Min trades/year = 20."),
        "geomean_year_score": ("score = geomean(year_pnl × min(year_pf, 3.0)) across all years. "
                                "Same per-year reset as default, but uses geometric mean instead of min. "
                                "Still punishes weak years on log-scale, but REWARDS strong years instead of ignoring them. "
                                "Fix for the W8 pattern where TPE found `barely-positive every year` strategies. "
                                "Any year with <=0 score -> hard penalty (preserves positivity gate)."),
        "activity_weighted": ("score = min(year_pnl × min(year_pf, 3.0)) × geomean(year_n) / 100. "
                               "Same as default but multiplied by activity bonus. "
                               "Rewards trials that trade actively across all years, not just clear the floor."),
        "hybrid_sum_minpf2": ("score = sum_pnl × min_pf². Keeps total-PnL signal but weights "
                                "minPF² hard. A trial with min_pf 0.3 scores 9× lower than min_pf 0.9 at same PnL. "
                                "Per-year min activity floor relaxed to 15 trades/year."),
    }
    return explanations.get(variant, str(variant))


def write_report(rounds):
    lines = []
    lines.append("# Autonomous multi-round Optuna orchestrator — full report\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append("")

    # ============ Story-level summary ============
    lines.append("## What this run did and why\n")
    lines.append(
        "Background: W7's empirical analysis (top 30 trials × 5 years backtested with fresh "
        "$25K each year) showed the legacy objective `total_pnl × min(pf, 3)` correlates only "
        f"0.281 (Pearson) with 2026 forward PnL. Per-year `min_pf` correlates 0.638 — over "
        "twice as informative. Compounding across training years was masking 2023's "
        "regime-fragility signal from TPE.\n"
    )
    lines.append(
        "The fix tested here: run the backtest in CV mode — reset cash to $25K at each "
        "year boundary so TPE sees per-year performance directly. Score by the WORST year's "
        "PnL × PF, forcing the optimizer to find configs that survive every regime, not just "
        "ace 2024-25.\n"
    )
    lines.append(
        "The orchestrator runs up to 3 rounds with different objective variants. If the "
        "default CV objective produces a 'boring strategy' (small trade counts hugging the floor) "
        "or a 'minPF-overfit' trial that wins train but flops forward, it auto-launches the next "
        "variant designed to fix that specific failure mode.\n"
    )
    lines.append(
        f"Baseline to beat: **W7 #254** had forward 2026 PnL = ${BASELINE_254_FORWARD:,} "
        "(8.98× starting cap). For context: the deployed **#124** baseline is +$22,534 (1.90×).\n"
    )

    # ============ TL;DR ============
    lines.append("## TL;DR\n")
    best_overall = None
    for r in rounds:
        if r["forwards"]:
            if best_overall is None or r["forwards"][0]["forward_pnl"] > best_overall[1]["forwards"][0]["forward_pnl"]:
                best_overall = (r["name"], r)
    if best_overall:
        name, r = best_overall
        w = r["forwards"][0]
        v_label, _ = r["verdict"]
        lines.append(f"- **Best round**: {name} (variant: `{r['variant'] or 'default'}`)")
        lines.append(f"  - Winner trial: **#{w['trial_number']}**")
        lines.append(f"  - Forward 2026 PnL: **${w['forward_pnl']:+,.0f}** ({w['forward_multi']:.2f}×)")
        lines.append(f"  - Max DD: {w['forward_max_dd_pct']:.1f}%   Trades: {w['forward_n_trades']}")
        delta_254 = w["forward_pnl"] - BASELINE_254_FORWARD
        delta_124 = w["forward_pnl"] - 22_534
        lines.append(f"  - vs W7 #254 baseline: ${delta_254:+,.0f}")
        lines.append(f"  - vs #124 deployed:    ${delta_124:+,.0f}")
        lines.append(f"  - Verdict: **{v_label}**")
    else:
        lines.append("- No round produced forward-testable results")
    lines.append("")

    # ============ Comparison vs baselines ============
    lines.append("## Comparison vs known baselines\n")
    lines.append("| Config | 2026 forward PnL | Multi | Trades | Note |")
    lines.append("|---|---|---|---|---|")
    lines.append("| #124 W3 deployed | +$22,534 | 1.90× | 196 | Currently live on $26K PKIPX |")
    lines.append("| #254 W7 (legacy obj) | +$199,553 | 8.98× | 215 | Forward winner of W7 — but no objective signal, just lucky |")
    for r in rounds:
        if r["forwards"]:
            w = r["forwards"][0]
            lines.append(f"| **{r['name']} winner #{w['trial_number']}** | ${w['forward_pnl']:+,.0f} | "
                          f"{w['forward_multi']:.2f}× | {w['forward_n_trades']} | "
                          f"variant: {r['variant'] or 'default'} |")
    lines.append("")

    # ============ Detailed per-round breakdown ============
    for r in rounds:
        lines.append(f"## Round {r['name']} — variant `{r['variant'] or 'default'}`\n")
        lines.append(f"**Objective formula**:\n  {_objective_explanation(r['variant'])}\n")
        s = r["state"]
        lines.append(f"**Trial counts**:")
        lines.append(f"- COMPLETE: {s['complete']}")
        lines.append(f"- RUNNING (at report time): {s['running']}")
        lines.append(f"- PRUNED (constraint violations): {s['pruned']}")
        lines.append("")

        # Top 5 forward
        if r["forwards"]:
            lines.append("**Top 5 ranked by 2026 forward PnL** (not by training score!):\n")
            lines.append("| Rank | Trial | Train score | Fwd PnL | Fwd multi | Fwd DD% | Fwd trades |")
            lines.append("|---|---|---|---|---|---|---|")
            for i, f in enumerate(r["forwards"]):
                lines.append(f"| {i+1} | #{f['trial_number']} | ${f['score']:,.0f} | "
                              f"${f['forward_pnl']:+,.0f} | {f['forward_multi']:.2f}× | "
                              f"{f['forward_max_dd_pct']:.1f}% | {f['forward_n_trades']} |")
            lines.append("")

            # Best-trial per-year detail
            w = r["forwards"][0]
            ua = w["user_attrs"]
            lines.append(f"**Per-year breakdown — winner #{w['trial_number']}** "
                          "(each year backtested with fresh $25K during training):\n")
            lines.append("| Year | PnL | PF | Trades |")
            lines.append("|---|---|---|---|")
            for y in ["2022","2023","2024","2025"]:
                lines.append(f"| {y} | ${ua.get(f'pnl_{y}','-')} | "
                              f"{ua.get(f'pf_{y}','-')} | {ua.get(f'n_{y}','-')} |")
            lines.append(f"\n- **min_pf** across years: **{ua.get('min_pf','-')}** "
                          "(the metric we optimized for)")
            lines.append(f"- **min_pnl** (worst year $ PnL): {ua.get('min_pnl','-')}")
            lines.append(f"- **sum_pnl** across years: {ua.get('sum_pnl','-')}")
            lines.append("")

        # Verdict
        verdict_label, next_variant = r["verdict"]
        lines.append(f"**Verdict for {r['name']}: `{verdict_label}`**\n")
        lines.append(f"_Why_: {_verdict_explanation(verdict_label, r['variant'])}\n")
        if next_variant:
            lines.append(f"_Action_: Next round will use variant `{next_variant}` because "
                          f"the {verdict_label} failure mode has a targeted fix.\n")

    # ============ 2023 evolution ============
    lines.append("## Did 2023 evolve across rounds? (the key validation signal)\n")
    lines.append("Background: every one of W7's top 30 trials LOST money on 2023 "
                  "(range −$2,608 to −$8,166). The hypothesis is that CV-style objectives "
                  "should produce trials that don't lose on 2023.\n")
    lines.append("| Round | Variant | Winner | 2023 PnL | min_pf | 2026 fwd PnL |")
    lines.append("|---|---|---|---|---|---|")
    lines.append("| W7 (reference) | legacy (compounded) | #254 | −$2,608 | 0.69 | +$199,553 |")
    for r in rounds:
        if not r["forwards"]: continue
        w = r["forwards"][0]
        ua = w["user_attrs"]
        lines.append(f"| {r['name']} | {r['variant'] or 'default'} | "
                      f"#{w['trial_number']} | "
                      f"{ua.get('pnl_2023','-')} | "
                      f"{ua.get('min_pf','-')} | "
                      f"${w['forward_pnl']:+,.0f} |")
    lines.append("")

    # ============ Decisions explained ============
    lines.append("## Methodology — every decision the orchestrator made\n")
    lines.append("**Decision tree:**\n")
    lines.append("```")
    lines.append("After each round:")
    lines.append("  IF winner_forward >= $150K           -> STOP, success")
    lines.append("  IF winner_forward < $80K")
    lines.append("    AND median(year_n) < 30            -> next variant = activity_weighted")
    lines.append("    AND min_pf > 0.7                   -> next variant = hybrid_sum_minpf2")
    lines.append("  ELSE                                 -> STOP, inconclusive (no clear remedy)")
    lines.append("Stop after 3 rounds or when all 2 alternate variants tried.")
    lines.append("```\n")
    lines.append("**Variants implemented:**\n")
    lines.append(f"- `default`: {_objective_explanation(None)}")
    lines.append(f"- `activity_weighted`: {_objective_explanation('activity_weighted')}")
    lines.append(f"- `hybrid_sum_minpf2`: {_objective_explanation('hybrid_sum_minpf2')}")
    lines.append("")

    # ============ What's next ============
    lines.append("## What to do next\n")
    if best_overall:
        name, r = best_overall
        w = r["forwards"][0]
        fwd = w["forward_pnl"]
        if fwd >= 150_000:
            lines.append(f"- {name} winner #{w['trial_number']} is a strong forward result. "
                          "Compare its trade composition (price distribution, slippage, "
                          "per-strategy contribution) against the deployed #124 before swapping.")
            lines.append("- Run live-vs-backtest audit on a few days to ensure modeled fills match.")
            lines.append("- Consider it as the next deployment candidate after the SIP upgrade.")
        elif fwd >= 80_000:
            lines.append(f"- {name} winner is competitive (+${fwd:,.0f}) but not a clear improvement "
                          f"over #254 (${BASELINE_254_FORWARD:,}). The CV objective is a methodology "
                          "win even if this run didn't produce a bigger number.")
            lines.append("- Consider running another round with a longer trial budget (1000+ trials) "
                          "or a smaller search space.")
        else:
            lines.append(f"- All variants tried produced sub-$80K forwards. The CV objective family "
                          "may not be the right fix — consider:")
            lines.append("  - Drop 2023 from training (it's only 67 microcap-pump days)")
            lines.append("  - Add 2021 to training (more data variety)")
            lines.append("  - Switch to true expanding walk-forward (train 22, test 23; train 22-23, test 24; etc.)")
    else:
        lines.append("- No round produced complete results; investigate worker logs.")
    lines.append("")

    # ============ Open question ============
    lines.append("## Caveat — known data gap\n")
    lines.append("- 2026 picks pkl has empty entries for 2026-05-21 through 2026-06-16 "
                  "(daily_top_gainers.csv has gappers, but picks builder produced empty lists). "
                  "All forward-2026 numbers above are effectively only **2026-01-05 through 2026-05-15**. "
                  "Picks rebuild deferred until after this orchestrator finishes "
                  "(would compete for CPU with the workers).")
    lines.append("")

    os.makedirs(os.path.dirname(FINAL_REPORT), exist_ok=True)
    with open(FINAL_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"Wrote {FINAL_REPORT}")


def main():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    log("=== Multi-round autonomous orchestrator started ===")

    results = []
    tried_variants = set()

    for i, round_cfg in enumerate(ROUNDS):
        # Pick variant for this round
        variant = round_cfg["variant"]
        if variant == "AUTO":
            if not results or not results[-1]["forwards"]:
                log("  No prior round result, stopping")
                break
            prev_verdict, prev_next = results[-1]["verdict"]
            if prev_verdict == "WIN":
                log("  Prior round WIN, no more rounds needed")
                break
            if prev_next is None:
                log("  Prior round INCONCLUSIVE, no clear next variant")
                # Try the unused variant anyway
                untried = ["activity_weighted", "hybrid_sum_minpf2"]
                untried = [v for v in untried if v not in tried_variants]
                if not untried:
                    log("  All variants tried, stopping")
                    break
                variant = untried[0]
            else:
                variant = prev_next
            log(f"  Auto-selected variant for {round_cfg['name']}: {variant}")

        if variant in tried_variants:
            log(f"  Variant {variant} already tried, skipping {round_cfg['name']}")
            continue
        if variant:
            tried_variants.add(variant)

        r = run_round(
            name=round_cfg["name"], db=round_cfg["db"], variant=variant,
            outdir=round_cfg["outdir"], log_subdir=round_cfg["log_subdir"],
            already_running=round_cfg["already_running"],
        )
        results.append(r)

        # Write incremental report after each round
        write_report(results)

        # USER OVERRIDE 2026-06-19: never stop early on WIN. Always run
        # all remaining variants so we can compare them all.
        verdict_label, _ = r["verdict"]
        if verdict_label == "WIN":
            log(f"  {r['name']} reached WIN threshold, but user requested all variants tested. Continuing.")

    write_report(results)
    log("\n=== Orchestrator complete ===")


if __name__ == "__main__":
    main()
