"""
Reads the differential-privacy telemetry written by FederatedTrainer
(artifacts/dp_report.json) so the dashboard can show REAL values
(per-hospital epsilon, Opacus noise multiplier, H, A, D, S) instead of typed ones.
"""
import json
import os
from typing import Any, Dict, Optional

DP_REPORT_PATH = "artifacts/dp_report.json"


def load_dp_report(path: str = DP_REPORT_PATH) -> Optional[Dict[str, Any]]:
    """Returns the saved report, or None if training has not produced one yet."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def latest_hospital_telemetry(report: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """{hospital_id: values} for the last training round, or {} if unavailable."""
    if not report or not report.get("rounds"):
        return {}
    return report["rounds"][-1].get("hospitals", {})
