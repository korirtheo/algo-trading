"""Check 7/17 P&L discrepancy between engine and Alpaca."""
import json
import sqlite3
from pathlib import Path

# Engine database
conn = sqlite3.connect("logs/trading.db")
cur = conn.cursor()

print("=" * 80)
print("ENGINE DATABASE (trading.db):")
print("=" * 80)

cur.execute("""
    SELECT ticker, strategy, entry_price, exit_price, shares, pnl, exit_time
    FROM trades
    WHERE DATE(exit_time) = '2026-07-17'
    ORDER BY exit_time
""")

db_trades = []
for row in cur.fetchall():
    ticker, strat, entry, exit_p, shares, pnl, exit_time = row
    db_trades.append({"ticker": ticker, "pnl": pnl})
    print(f"{ticker:6} ({strat}): ${entry:.2f} x {shares} -> ${exit_p:.2f} = ${pnl:,.2f}")

db_total = sum(t["pnl"] for t in db_trades)
print(f"\nDatabase Total P&L: ${db_total:,.2f}")

conn.close()

# Engine JSON trades file
print("\n" + "=" * 80)
print("ENGINE JSON LOG (2026-07-17_trades.json):")
print("=" * 80)

trades_file = Path("logs/2026-07-17_trades.json")
if trades_file.exists():
    with open(trades_file) as f:
        trades = json.load(f)

    json_total = 0
    for t in trades:
        print(f"{t['ticker']:6} ({t['strategy']}): entry=${t['entry_price']:.2f} x {t['shares']} -> ${t['exit_price']:.2f} (reason={t['reason']})")
        print(f"  Peak: ${t['peak_price']:.2f}, Trail: {t['trail_pct']}%")
        print(f"  P&L: ${t['pnl']:,.2f} ({t['pnl_pct']:.2f}%)")
        json_total += t["pnl"]

    print(f"\nJSON Total P&L: ${json_total:,.2f}")
else:
    print("No JSON log found")

print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)
print(f"Database: ${db_total:,.2f}")
print(f"JSON: ${json_total:,.2f}")
print(f"Match: {db_total == json_total}")

print("\nTo get Alpaca's actual fill prices, need to check:")
print("  1. SDOT: what were the actual buy/sell fill prices on Alpaca?")
print("  2. VEEE: what were the actual buy/sell fill prices on Alpaca?")
print("\nEngine is calculating P&L based on simulated fills, not actual Alpaca fills.")
