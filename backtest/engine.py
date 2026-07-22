from test_green_candle_combined import simulate_day_combined as _simulate_day_combined


def simulate_day_combined(
    picks, cash, cash_account=False, is_live=False, params=None, real_time_bar=None
):
    """Simulate combined strategy for one day with single cash pool."""
    return _simulate_day_combined(picks, cash, cash_account, is_live, params)
