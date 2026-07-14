@echo off
python -c "import py_compile; py_compile.compile('live/engine_combined.py', doraise=True); print('Compile: OK')"
python scripts/analysis/backtest_511_plus_v3_2026.py
pause
