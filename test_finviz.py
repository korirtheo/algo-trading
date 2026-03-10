import requests
from io import StringIO
import pandas as pd

headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0"}
resp = requests.get(
    "https://finviz.com/screener.ashx",
    headers=headers,
    params={"v": "111", "f": "sh_price_u50", "o": "-prechange"},
    timeout=15,
)
print("Status:", resp.status_code)
tables = pd.read_html(StringIO(resp.text))
for i, t in enumerate(tables):
    if "Ticker" in [str(c).strip() for c in t.columns]:
        print(f"Table {i} cols:", list(t.columns))
        print(t.head(5).to_string())
        break
