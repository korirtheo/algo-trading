import pandas as pd
from research.legacy_long_resurrection.causal_execution import Signal, next_bar_open_fill, assert_causal_fill


def _bars():
    idx=pd.date_range("2026-01-02 09:30", periods=4, freq="2min", tz="America/New_York")
    return pd.DataFrame({"Open":[10,11,12,13],"High":[11,12,13,14],"Low":[9,10,11,12],
                         "Close":[10.5,11.5,12.5,13.5],"Volume":[100,200,300,400]}, index=idx)


def test_close_signal_fills_next_bar_open():
    b=_bars(); s=Signal("TEST",1,b.index[1],float(b.iloc[1].Close),"CONFIRM")
    f=next_bar_open_fill(s,b,0.3)
    assert f.fill_bar_index == 2
    assert f.raw_fill_price == 12.0
    assert f.fill_price == 12.0*1.003
    assert_causal_fill(f)


def test_last_bar_signal_cannot_fill():
    b=_bars(); s=Signal("TEST",3,b.index[3],13.5,"CONFIRM")
    assert next_bar_open_fill(s,b,0.3) is None


def test_stale_prior_price_is_never_used():
    b=_bars(); s=Signal("TEST",1,b.index[1],9.99,"CONFIRM")
    f=next_bar_open_fill(s,b,0.0)
    assert f.raw_fill_price == float(b.iloc[2].Open)
    assert f.raw_fill_price != s.reference_price
