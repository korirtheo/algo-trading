"""
Download Q4 2024 (Oct-Dec) data using a SEPARATE Polygon API key.
Runs in parallel with other downloads without rate-limit conflicts.

Usage:
  python download_2024_q4.py
"""
import os
import sys
import subprocess

API_KEY = "6VlSpO94hHOpcQ5XOEe9clXEEpbdyaGb"
OUT_DIR = "stored_data_oct_dec_2024"
START = "2024-10-01"
END = "2024-12-31"

print(f"{'='*70}")
print(f"  Q4 2024 DOWNLOAD (separate API key)")
print(f"  Period: {START} to {END} -> {OUT_DIR}/")
print(f"{'='*70}\n")

env = os.environ.copy()
env["POLYGON_OUT_DIR"] = OUT_DIR

result = subprocess.run(
    [sys.executable, "download_polygon.py", API_KEY, START, END],
    env=env,
)

if result.returncode != 0:
    print(f"\nERROR: Q4 download failed with exit code {result.returncode}")
    sys.exit(1)

print(f"\nQ4 2024 download complete -> {OUT_DIR}/")
