import yfinance as yf
import pandas as pd
import datetime

# 1. Define tickers to scan (or pull from an index)
tickers = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOG"]

# 2. Define backtest period
start_date = "2024-03-25"
end_date = datetime.datetime.today().strftime("%Y-%m-%d")

# 3. Collect historical data
data_dict = {}
for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date)
    if not df.empty:
        df["PctChange"] = df["Close"].pct_change() * 100
        data_dict[ticker] = df

# 4. Backtest: find top gainer each day
results = []
for date in pd.date_range(start=start_date, end=end_date):
    day_gainers = []
    for ticker, df in data_dict.items():
        if date in df.index:
            day_gainers.append(
                (ticker, df.loc[date, "PctChange"], df.loc[date, "Close"])
            )
    if day_gainers:
        # Sort by % gain
        day_gainers.sort(key=lambda x: x[1], reverse=True)
        top = day_gainers[0]
        results.append(
            {
                "Date": date.date(),
                "Ticker": top[0],
                "PctChange": top[1],
                "Close": top[2],
            }
        )

# Convert to DataFrame
results_df = pd.DataFrame(results)
print("Top gainers per day:")
print(results_df)

# Optional: plot cumulative returns if you 'buy top gainer' each day
results_df["Return"] = results_df["PctChange"] / 100
results_df["Cumulative"] = (1 + results_df["Return"]).cumprod()
results_df.set_index("Date", inplace=True)
results_df["Cumulative"].plot(title="Cumulative Performance if Buying Top Gainer Daily")
