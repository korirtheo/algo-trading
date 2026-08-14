# AGENTS.md

## AWS Deployment — GOLDEN RULE

**Always deploy via git: commit + push locally, then `git pull` on AWS. NEVER edit files directly on the AWS box (no scp, no sed, no manual file writes there).**

The only file edited directly on the server is `.env` (keys + `LIVE_PARAMS_PATH`) — and even that is just flipping pointers / secret values, never editing code/config logic.

## Secrets (.env-based, since 2026-08-13)

- **API keys are NEVER in the repo.** `config/settings.py` requires them from the environment — no hardcoded fallbacks. Keys load from the repo-root `.env` (gitignored) via `dotenv`.
- Keys required: `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `TRADIER_API_KEY` (Polygon keys optional, downloads only: `POLYGON_API_KEY[_2/_3/_4]`).
- Template: `.env.example` (committed). Never commit `.env`.
- **Every environment** (local + each AWS server) has its own `.env`. `docker-compose.yml` passes `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `TRADIER_API_KEY`, `ALPACA_PAPER`, `LIVE_PARAMS_PATH` into the container.
- **Rotating keys**: update local `.env` → commit NOTHING secret → `git pull` on AWS → rewrite server `.env` → `sudo docker compose build && sudo docker compose up -d`. Without keys in `.env`, the container will not start (settings raises at import).

## Live bot

- Runs on AWS EC2 (Ubuntu) as a **Docker container** (`algotrader`, image `algo-trading-algotrader`), managed by `docker-compose.yml` in the repo root.
- SSH: `ssh -i "C:\Users\Theo Korir\Documents\Python\algo-trading\trading-key-v2.pem" ubuntu@54.172.65.25`
- Repo on server: `/home/ubuntu/algo-trading`
- `config/` and `logs/` are **bind-mounted** into the container (`./config:/app/config`, `./logs:/app/logs`) — config and DB state persist across container recreates. **Code is baked into the image** → code changes require a rebuild.
- `LIVE_PARAMS_PATH` (in `.env`) selects the deploy config; the engine's `load_trial_params` reads it at startup.

## Deploy sequence (code changes)

```bash
# local
git add <files> && git commit -m "..." && git push origin main

# aws
ssh -i "trading-key-v2.pem" ubuntu@54.172.65.25
cd /home/ubuntu/algo-trading && git pull origin main
sudo docker compose build
sudo docker compose up -d
```

## Deploy sequence (config-only change, e.g. switching LIVE_PARAMS_PATH)

No image rebuild needed (config is bind-mounted):
```bash
# commit + push the config locally, git pull on aws, then:
cd /home/ubuntu/algo-trading
sed -i 's|LIVE_PARAMS_PATH=.*|LIVE_PARAMS_PATH=config/<name>.json|' .env
sudo docker compose up -d
```

## Key rotation / new-account deploy

1. Get new keys, write them to **local** `.env` (never commit).
2. `git pull` on AWS (gets any new code).
3. Rewrite the **server** `.env` with the new keys (this is the one allowed direct server edit).
4. `sudo docker compose build && sudo docker compose up -d` — rebuild required because `config/settings.py` is baked into the image and now raises if env keys are missing.

## Deployed bot facts (2026-08-13)

- Currently: **G-only 1x** — `config/trial_gl_1min_g2_1x_G_only_deploy.json` (G2 #106 first-bar-only, L #312 disabled while its winner-filter analysis continues).
- **2026-08-13: Alpaca + Tradier keys rotated** and moved out of the repo into gitignored `.env` (both local and server). Old keys were committed in history — revoke/rotate them at the provider if they're still valid.
- `strategies/bars.py` was missing from the deployed image (root-caused 2026-08-12: `No module named 'strategies.bars'`, every bar errored, no trades). It is committed; keep it in the repo.
- Multi-window slippage + vol caps are active live (`LIVE_DISABLE_VOL_CAPS=False`); module defaults match the OOS runner (K=3.0, 15%/8%/5% caps).

## OOS conventions

- 1-min data cache: `stored_data_1min/fulltest_picks_gap2_vol250k.pkl` (picks keyed by date).
- Deploy OOS window: **2026-03-01 → 2026-08-07 (~100 trading days)**; full-sample runs use 2024-01-01 → 2026-12-31.
- Run OOS with the deploy config top-level fields applied exactly like `load_trial_params`: `g_first_bar_only`, `margin_multiplier`, `max_position_pct_of_cash`, and `l_filter*` → `tgc.L_FILTER_*`. `set_strategy_params(cfg['params'])` alone misses these.

## Optuna conventions

- **ALWAYS use PostgreSQL for the Optuna study DB** — never SQLite for a real run. Postgres 16/17 runs locally at `postgresql://postgres@127.0.0.1:5432/` (trust auth, no password). One **database per study family** (e.g. `optuna_gl_1min`, `optuna_g_wide`, `optuna_g_only`), study name = the variant (e.g. `gl_1min_v5_g2_2x`, `g_wide_w21b`). Pass via `--db postgresql://postgres@127.0.0.1:5432/<db> --study <name>`. SQLite (`--db *.db`) is only for throwaway smoke tests.
- **Optimal startup trials for TPE**: use `--startup-trials 4x` the number of tunable dimensions (a ~13-param exit study → `--startup-trials 50-80`). The default 200 is for the 20-strategy combined space; scale down for focused single-strategy studies.
- **Slippage parity is mandatory**: the objectives assert `USE_DYNAMIC_SLIPPAGE` + `USE_MULTIWINDOW_SLIPPAGE` are True. Never disable for a study.
- G exit-shape study (2026-08-14): entry locked from `config/trial_gl_1min_g2_1x_G_only_deploy.json`, only exit params tunable, via env `G_FIXED_FROM_CONFIG=<config> G_TUNE_EXITS_ONLY=1 ALLOWED_STRATS=g`. Tunable exit vars: `g_target_pct`, `g_target2_pct`, `g_partial_sell_pct`, `g_time_limit_min`, `g_stop_pct`, `g_trail_pct`, `g_trail_activate_pct`, `g_exit_mode` (fixed/atr/swing/staged), `g_exit_atr_mult`, `g_exit_swing_k`, `g_exit_swing_window`, `g_exit_staged_thresh`, `g_exit_staged_wide`.
- Best-trial params go to `results/params/` (e.g. `--params-out results/params/<study>_best.json`), then validate OOS before deploying.

## Exit-strategy evaluation — STANDING RULE (2026-08-14, after STKH/G exit study)

**Never evaluate an exit-strategy candidate on compounded $ PnL alone.** Always pair
it with per-trade mean/median AND a bootstrap CI before treating a $ result as signal.

- Why: compounding math + fat right tails (G MFE p95 +142%) can make a LOSING
  per-trade config look like a winner on $ PnL over a short window. Example: #101
  looked +20-24% on 2025/2026 OOS $ PnL but was −1.3%/trade (hand-rolled sim) or
  +0.3% inconclusive (real engine) on the aligned full-sample bootstrap.
- Protocol: (1) run the candidate AND baseline through the REAL
  `simulate_day_combined` (slippage ON, volcaps ON, deploy entry gates applied);
  (2) align on the common (date,ticker) trade set; (3) bootstrap the per-trade
  edge (10k resamples); (4) only claim an edge if the CI excludes 0. If CI
  straddles 0, the result is INCONCLUSIVE — say so, don't call it a win.
- Also: hold out a genuinely-unused year (e.g. 2025) BEFORE trusting any "edge"
  found on the search window; check for degenerate basins (e.g. stop=0 W16 trap);
  and don't trust single-best-trial selection — use wide-forward distributions.
- The deployed G baseline (1% trail, tgt +18, stop 10, no partial) is currently
  the only config supported by per-trade bootstrap evidence. Full study:
  `results/g_exit_investigation_summary.md`.
