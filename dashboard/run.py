"""
Standalone dashboard server for development/testing.
Connects to Alpaca API for account data but no live engine.

Usage:
  python -m dashboard.run           # dev mode on port 8000
  python -m dashboard.run --port 3001
"""
import argparse
import uvicorn
from dashboard.backend.app import app


def main():
    parser = argparse.ArgumentParser(description="AlgoTrader Dashboard (standalone)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print(f"Dashboard: http://localhost:{args.port}")
    print("Note: Running standalone — no live engine connected.")
    print("      Use 'python -m live.main' for live trading + dashboard.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
