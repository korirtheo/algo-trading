"""
Download Q3 2024 (Jul-Sep) data using a SEPARATE Polygon API key.
Runs in parallel with other downloads without rate-limit conflicts.

Usage:
  python download_2024_q3.py
"""
import os
import sys
import subprocess

API_KEY = "dE5ScpEiJH3M5slg3pFjcC7tkL1b4JXI"
OUT_DIR = "stored_data_jul_sep_2024"
START = "2024-07-01"
END = "2024-09-30"

print(f"{'='*70}")
print(f"  Q3 2024 DOWNLOAD (separate API key)")
print(f"  Period: {START} to {END} -> {OUT_DIR}/")
print(f"{'='*70}\n")

env = os.environ.copy()
env["POLYGON_OUT_DIR"] = OUT_DIR

result = subprocess.run(
    [sys.executable, "download_polygon.py", API_KEY, START, END],
    env=env,
)

if result.returncode != 0:
    print(f"\nERROR: Q3 download failed with exit code {result.returncode}")
    sys.exit(1)

print(f"\nQ3 2024 download complete -> {OUT_DIR}/")
