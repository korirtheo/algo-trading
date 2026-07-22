import requests

api_key = "o9vwR8VA1nPulvAo9A5Dk8FGvoJV"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Accept": "application/json",
}

url = "https://api.tradier.com/v1/markets/timesales"
params = {
    "symbol": "AAPL",
    "interval": "1min",
    "start": "2026-07-21 09:30",
    "end": "2026-07-21 16:00",
}

resp = requests.get(url, params=params, headers=headers)
print("Status Code:", resp.status_code)
try:
    print(resp.json())
except Exception as e:
    print(resp.text)
