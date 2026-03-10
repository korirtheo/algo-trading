import requests
from io import StringIO
import pandas as pd

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}
resp = requests.get(
    "https://finviz.com/screener.ashx",
    headers=headers,
    params={"v": "111", "f": "sh_price_u50", "o": "-change"},
    timeout=15,
)
print("Status:", resp.status_code)
print("Response size:", len(resp.text), "bytes")

tables = pd.read_html(StringIO(resp.text))
print(f"Total tables: {len(tables)}")
print()

for i, t in enumerate(tables):
    cols = [str(c).strip() for c in t.columns]
    has_ticker = "Ticker" in cols
    has_price = "Price" in cols
    has_change = "Change" in cols
    marker = " <-- MATCH" if (has_ticker and has_price and has_change) else ""
    print(f"Table {i}: {len(t)} rows x {len(cols)} cols | Ticker={has_ticker} Price={has_price} Change={has_change}{marker}")
    print(f"  cols: {cols[:12]}")
    if has_ticker and has_price:
        print(f"  first 3 tickers: {list(t['Ticker'][:3])}")
    print()
