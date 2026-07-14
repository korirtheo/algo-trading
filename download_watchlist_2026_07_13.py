"""
Download 2026-07-13 data for the live watchlist (ELAB, VEEE).
"""
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET
import os

client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

# The two tickers that were attempted on 2026-07-13
watchlist = ['ELAB', 'VEEE']
target = datetime(2026, 7, 13)
prev_date = datetime(2026, 7, 11)  # Friday before Monday 7/13

data_dir = 'stored_data_jul_2026'
os.makedirs(f'{data_dir}/intraday', exist_ok=True)
os.makedirs(f'{data_dir}/daily', exist_ok=True)

print(f'Downloading 2026-07-13 data for watchlist: {watchlist}\n')

for ticker in watchlist:
    print(f'[{ticker}]')

    # Get prev close
    try:
        prev_req = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame.Day,
            start=prev_date,
            end=prev_date + timedelta(days=1),
            limit=1
        )
        prev_bars = client.get_stock_bars(prev_req)
        if ticker in prev_bars:
            prev_df = prev_bars[ticker].df
            prev_df.to_csv(f'{data_dir}/daily/{ticker}.csv')
            print(f'  Daily: {len(prev_df)} bars, prev_close=${prev_df.iloc[-1]["close"]:.2f}')
    except Exception as e:
        print(f'  Daily: ERROR - {e}')

    # Get intraday
    try:
        intra_req = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame.Minute,
            start=target,
            end=target + timedelta(days=1),
            limit=1000
        )
        intra_bars = client.get_stock_bars(intra_req)
        if ticker in intra_bars:
            intra_df = intra_bars[ticker].df
            intra_df.to_csv(f'{data_dir}/intraday/{ticker}.csv')
            print(f'  Intraday: {len(intra_df)} bars')
            print(f'  First bar: {intra_df.index[0]}')
            print(f'  Open: ${intra_df.iloc[0]["open"]:.2f}')
    except Exception as e:
        print(f'  Intraday: ERROR - {e}')

    print()

print(f'Download complete. Files saved to {data_dir}/')
