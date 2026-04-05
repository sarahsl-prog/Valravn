from __future__ import annotations


def check_anomalies(state: dict) -> dict:
    return {"_pending_anomalies": False}


def record_anomaly(state: dict) -> dict:
    return {}
