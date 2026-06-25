# Engine Multi-Position Spec — adding v3 R-O alongside G+L

**Goal**: enable `live/engine_combined.py` to hold up to N concurrent positions across different strategies on the same account, so v3 R-O can run alongside G+L without blocking each other.

**Expected lift**: +$1.49M / +40.6% over #511 baseline on 2026 backtest (see `results/backtest_511_plus_v3_2026.json`).

---

## 1. Current state — what blocks v3

`live/engine_combined.py:144` — `self.active_position = None` is a **single ticker slot**:

```python
self.active_position = None  # ticker currently in position
self.position_entry = {}     # {ticker: {entry_price, shares, strategy, entry_time, cost}}
```

Signal logic enforces the single-slot invariant (~line 413, 442, 476-477):
- If `ticker == self.active_position` → handle as own ticker
- Elif `self.active_position is None` → fire signal, claim slot
- Else → skip (`SIGNAL X skipped — already in position Y`)

This makes G+L+v3 deployment impossible — when G holds, L and v3 both skip; when L holds, G and v3 skip.

## 2. Target architecture — per-strategy slots

### Data model
```python
# Replace single ticker slot with per-strategy dict of held tickers
self.active_positions = {}  # {strategy_letter: ticker} e.g. {"G": "KUST", "L": "MIMI"}

# position_entry remains keyed by ticker (one position per ticker still)
self.position_entry = {}    # {ticker: {entry_price, shares, strategy, entry_time, cost}}
```

### Strategy slot rules
- Each strategy gets ONE slot. With G, L, R-O enabled → 3 slots → up to 3 concurrent positions.
- Each strategy can only hold ONE ticker at a time (its slot).
- Different strategies CAN hold the same ticker at different times (e.g. G held KUST and exited; later R-O can buy KUST). This was the user's design call.
- Same-bar same-ticker conflict (rare but possible): resolved by `STRAT_PRIORITY` dict. Lower number wins. Suggested addition: `"R-O": 2` (just below G=1).

### Capital
- Each trade sizes at `30% of CURRENT cash` (existing behavior, unchanged).
- With 3 slots filled, up to ~90% of cash is deployed. Cash floor stays at 0 (never negative).
- Vol caps + slippage applied per-trade (unchanged).

## 3. Implementation phases

### Phase 1 — refactor data structures (no behavior change)
Replace single `active_position` with `active_positions: dict[str, str|None]` keyed by strategy letter. Initialize each enabled strategy with `None`. Keep `position_entry` keyed by ticker.

**Files**: `live/engine_combined.py` only.

**Verify**: existing single-strategy flows still work. All references to `self.active_position` updated to use the dict lookup.

### Phase 2 — signal/reconcile/exit logic per slot
Update three sites:

1. **Signal handler** (~line 413-442, 476-477):
   - Old: `if ticker == self.active_position` / `elif self.active_position is None`
   - New: `if ticker == self.active_positions.get(strat)` / `elif self.active_positions.get(strat) is None`
   - The skip message changes from "already in position Y" to "strategy X slot held"

2. **Reconcile logic** (~line 342-393, fixed earlier today):
   - Iterate per strategy slot: `for strat, ticker_pos in self.active_positions.items():`
   - "No position at Alpaca + no pending buy" → clear that strat's slot only, not all
   - Reconcile pending-buy check (today's fix) preserved per-slot

3. **Sell/exit flow** (~line 713-714, 778, 794):
   - On fill confirmation, clear the strategy slot that opened the position, not all slots
   - Use `self.position_entry[ticker]["strategy"]` to know which slot to clear

### Phase 3 — add v3 R-O as a strategy in the engine
Currently v3 lives outside the simulator (we layered it externally in backtests). For live deployment v3 needs to be a first-class strategy:

- Add `enable_x` flag for R-O (let's call it strategy "X" — already in the priority dict; or pick a fresh letter)
- Add v3 entry rule logic in the signal loop:
  - Skip bar 0
  - At bar 1+: if bar 0 RED → first close > day_open fires entry
  - If bar 0 GREEN + G holds ticker → wait until G exits, then scan
  - Else → first close > day_open from bar 1+
- Add v3 exit rule: same target/stop/time/trail params as #576 (target=57, stop=30, time=27, trail=0.5/act=0)
- Hook into `set_strategy_params` so live config carries v3 params

**Files**: `test_green_candle_combined.py` (add v3 logic to `simulate_day_combined`), `optimize_combined.py` (param wiring), `live/engine_combined.py` (slot enablement).

Note: v3 currently shares strategy letter "G" naming in our R-O Optuna scripts (`g_target_pct`, etc. as param keys). Rename to v3-specific keys for live (e.g. `ro_target_pct`, `ro_stop_pct`).

### Phase 4 — same-bar priority routing
Update `STRAT_PRIORITY` dict to include R-O. When multiple strategies want to enter the **same ticker** on the same bar, priority wins. (Same-strategy same-ticker is already a no-op via slot check.)

Different tickers on same bar: no conflict, both fire if both slots free.

## 4. Testing strategy

### Pre-merge
- Unit test: 3 slots fire on 3 different tickers same day → 3 positions held
- Unit test: G holds, R-O fires on same ticker after G exits → R-O takes new position
- Unit test: Both G and L lose their tickers to brackets, both slots clear independently
- Integration test: 2026 backtest with v3-as-native vs externally-layered v3 should produce **same PnL within slippage tolerance** ← critical regression check
- Replay test: feed yesterday's bar data through engine with new code, verify same trade decisions
- Live dry-run: 1 trading day with `--dry-run`, verify all 3 strategies signal correctly without placing orders

### Post-deploy
- Watch first day with multi-position closely
- Compare live trades to backtest projection for same day
- Roll back to single-position if anomalies

## 5. Rollback plan

Single config flag `ENABLE_MULTI_POSITION = False` reverts to legacy single-slot behavior. If multi-position causes issues:
1. Set `ENABLE_MULTI_POSITION=False` in environment
2. Restart Docker
3. Engine falls back to single-slot, G+L only (current production)
4. Open positions held overnight are protected by Alpaca brackets regardless

## 6. Estimated effort

- Phase 1 (data structures): 2-3 hours
- Phase 2 (signal/reconcile/exit logic): 4-6 hours
- Phase 3 (v3 as native strategy): 4-6 hours (most involved — needs simulator changes too)
- Phase 4 (priority routing): 1-2 hours
- Testing: 4-6 hours

**Total: 2-3 days of focused work** before live deploy. Significantly less if Phase 3 can keep v3 external (layered) and only the slot count is changed.

## 7. Alternative: lightweight version (just multi-slot, skip native v3)

If we don't want to add v3 to the simulator yet:
- Phase 1 + 2 + 4 only (~1-2 days)
- v3 remains external: a separate process or script that watches the same account and places R-O orders independently
- Multi-position support lets v3's orders coexist with G+L orders on the same account
- Pros: faster to ship, isolated v3 bugs don't affect G+L
- Cons: two processes managing one account (race conditions on order submission; capital tracking duplicated)

Recommendation: do the lightweight version first to capture the v3 alpha quickly, then move v3 inside as Phase 3 once the multi-slot machinery is stable.
