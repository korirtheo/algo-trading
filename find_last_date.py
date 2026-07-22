
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_full import load_all_picks

data_dirs = ['stored_data', 'stored_data_mar_may_2026', 'stored_data_jun_2026', 'stored_data_jul_2026']
all_dates, _ = load_all_picks(data_dirs)
print(f"Last date: {all_dates[-1]}")
