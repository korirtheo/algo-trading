"""
Download all of 2024 data from Polygon.io in quarterly chunks.
Runs sequentially to stay within free-plan rate limits (5 calls/min).

Usage:
  python download_2024.py
"""
import os
import sys
import subprocess

API_KEY = "XKAw9xOfkdhbNT9iKpZit_npwf010c8q"

QUARTERS = [
    ("stored_data_jan_mar_2024", "2024-01-01", "2024-03-31"),
    ("stored_data_apr_jun_2024", "2024-04-01", "2024-06-30"),
    ("stored_data_jul_sep_2024", "2024-07-01", "2024-09-30"),
    ("stored_data_oct_dec_2024", "2024-10-01", "2024-12-31"),
]

for i, (out_dir, start, end) in enumerate(QUARTERS, 1):
    print(f"\n{'='*70}")
    print(f"  QUARTER {i}/4: {start} to {end} -> {out_dir}/")
    print(f"{'='*70}\n")

    env = os.environ.copy()
    env["POLYGON_OUT_DIR"] = out_dir

    result = subprocess.run(
        [sys.executable, "download_polygon.py", API_KEY, start, end],
        env=env,
    )

    if result.returncode != 0:
        print(f"\n  ERROR: Quarter {i} failed with exit code {result.returncode}")
        print(f"  You can resume by editing QUARTERS list to skip completed ones.")
        sys.exit(1)

    print(f"\n  Quarter {i}/4 complete!")

print(f"\n{'='*70}")
print(f"  ALL 2024 DATA DOWNLOADED SUCCESSFULLY")
print(f"{'='*70}")
