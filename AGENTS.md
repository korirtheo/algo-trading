# AGENTS.md

## AWS Deployment — GOLDEN RULE

**Always deploy via git: commit + push locally, then `git pull` on AWS. NEVER edit files directly on the AWS box (no scp, no sed, no manual file writes there).**

The only file edited directly on the server is `.env` (to switch `LIVE_PARAMS_PATH` between deploy configs) — and even that is just flipping a pointer, never editing code/config logic.

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

## Deployed bot facts (2026-08-12)

- Currently: **G-only 1x** — `config/trial_gl_1min_g2_1x_G_only_deploy.json` (G2 #106 first-bar-only, L #312 disabled while its winner-filter analysis continues).
- `strategies/bars.py` was missing from the deployed image (root-caused 2026-08-12: `No module named 'strategies.bars'`, every bar errored, no trades). It is committed; keep it in the repo.
- Multi-window slippage + vol caps are active live (`LIVE_DISABLE_VOL_CAPS=False`); module defaults match the OOS runner (K=3.0, 15%/8%/5% caps).

## OOS conventions

- 1-min data cache: `stored_data_1min/fulltest_picks_gap2_vol250k.pkl` (picks keyed by date).
- Deploy OOS window: **2026-03-01 → 2026-08-07 (~100 trading days)**; full-sample runs use 2024-01-01 → 2026-12-31.
- Run OOS with the deploy config top-level fields applied exactly like `load_trial_params`: `g_first_bar_only`, `margin_multiplier`, `max_position_pct_of_cash`, and `l_filter*` → `tgc.L_FILTER_*`. `set_strategy_params(cfg['params'])` alone misses these.
