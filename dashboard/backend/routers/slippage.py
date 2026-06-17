"""Slippage calibration endpoints — read logs/fills_calibration.csv."""
import os
from fastapi import APIRouter

router = APIRouter(tags=["slippage"])


@router.get("/slippage/recent")
async def get_slippage_recent(limit: int = 100):
    """Return the most recent N fills with their slippage detail + aggregates."""
    from dashboard.backend.app import bridge
    return bridge.get_slippage_data(limit=limit)


@router.get("/slippage/health")
async def get_slippage_health():
    """Diagnose why /slippage/recent might be returning empty data.

    Reports:
      - whether logs/fills_calibration.csv exists at the expected path
      - file size + row count + last-modified
      - the resolved absolute path the bridge is reading from

    If the dashboard shows 'no data', hit this endpoint to see if it's:
      (a) file missing       -> executor never wrote (calibration broken)
      (b) file empty/header  -> no terminal fills yet OR _reconcile_fill_async dying
      (c) path mismatch      -> dashboard reading different dir than executor
    """
    # File is slippage.py at dashboard/backend/routers/slippage.py
    # so we need 4 dirname()s to reach the project root, then logs/...
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))),
        "logs", "fills_calibration.csv",
    )
    info = {
        "expected_path": path,
        "exists": os.path.exists(path),
        "size_bytes": None,
        "row_count": None,
        "header_only": None,
        "last_modified": None,
        "logs_dir_exists": os.path.exists(os.path.dirname(path)),
        "logs_dir_contents": [],
    }
    try:
        if info["logs_dir_exists"]:
            info["logs_dir_contents"] = sorted(os.listdir(os.path.dirname(path)))[:50]
    except Exception as e:
        info["logs_dir_error"] = str(e)

    if info["exists"]:
        try:
            st = os.stat(path)
            info["size_bytes"] = st.st_size
            info["last_modified"] = st.st_mtime
            with open(path) as f:
                lines = f.readlines()
            info["row_count"] = max(0, len(lines) - 1)
            info["header_only"] = (len(lines) <= 1)
        except Exception as e:
            info["read_error"] = str(e)

    # Verdict
    if not info["exists"]:
        info["verdict"] = "file_missing"
        info["explanation"] = (
            "logs/fills_calibration.csv does not exist. The executor's "
            "_ensure_fill_log_header() should create it on OrderExecutor init. "
            "Either the executor hasn't initialized in this container, the "
            "logs/ volume mount is misconfigured, or write permissions failed."
        )
    elif info["header_only"]:
        info["verdict"] = "no_calibration_rows"
        info["explanation"] = (
            "File exists with header but no data rows. Either no orders have "
            "reached terminal status yet, or _reconcile_fill_async background "
            "threads are dying silently. Check live engine logs for 'reconcile' "
            "warnings."
        )
    elif info["row_count"] and info["row_count"] > 0:
        info["verdict"] = "ok"
        info["explanation"] = (
            f"{info['row_count']} calibration rows present. If the dashboard "
            f"still shows no data, the bug is in the frontend rendering."
        )
    else:
        info["verdict"] = "unknown"

    return info
