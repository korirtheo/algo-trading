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
    params={"v": "111", "f": "sh_price_u50", "o": "-prechange"},
    timeout=15,
)
print("Status:", resp.status_code)
print("Response size:", len(resp.text), "bytes")

tables = pd.read_html(StringIO(resp.text))
print(f"Total tables found: {len(tables)}")
print()

for i, t in enumerate(tables):
    cols = [str(c).strip() for c in t.columns]
    print(f"Table {i}: {len(t)} rows, cols={cols}")
    if "Ticker" in cols:
        print(f"  *** HAS TICKER — also has Price={('Price' in cols)}, Change={('Change' in cols)}")
        print(t.head(5).to_string())
    print()
