from __future__ import annotations

import json
import os

import requests

ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")
TASKS = ["orders", "user-merge", "transactions"]


def call_env(endpoint: str, method: str = "POST", data: dict | None = None, params: dict | None = None) -> dict:
    url = f"{ENV_URL}/{endpoint}"
    if method == "POST":
        response = requests.post(url, json=data, params=params, timeout=45)
    else:
        response = requests.get(url, params=params, timeout=45)
    response.raise_for_status()
    return response.json()


def pick_action(task_id: str, obs: dict) -> dict:
    summary = obs["error_summary"]

    if task_id == "orders":
        if summary["nulls"] > 0:
            return {"action_type": "fill_nulls", "column": "amount_usd", "params": {"strategy": "value", "value": 0}}
        if summary["duplicates"] > 0:
            return {"action_type": "drop_duplicates", "params": {"key_cols": ["order_id"], "keep": "first"}}
        if summary["format_violations"] > 0:
            return {"action_type": "cast_column", "column": "order_date", "params": {"target_type": "datetime"}}
        if summary["outliers"] > 0:
            return {"action_type": "clip_outliers", "column": "amount_usd", "params": {"method": "zscore", "bounds": [0, 10000]}}
        return {
            "action_type": "normalize_values",
            "column": "status",
            "params": {"mapping": {"SHIPPED": "shipped", "PENDING": "pending", "DELIVERED": "delivered", "CANCELLED": "cancelled"}},
        }

    if task_id == "user-merge":
        if summary["duplicates"] > 0:
            return {"action_type": "drop_duplicates", "params": {"key_cols": ["email"], "keep": "first"}}
        if summary["nulls"] > 0:
            for col, default in [
                ("name", "Unknown User"),
                ("phone", "+1-000-0000"),
                ("event_ts", "2025-01-01T00:00:00Z"),
                ("event_name", "unknown"),
            ]:
                if obs["null_counts"].get(col, 0) > 0:
                    return {"action_type": "fill_nulls", "column": col, "params": {"strategy": "value", "value": default}}
        if summary["format_violations"] > 0:
            return {"action_type": "normalize_values", "column": "phone", "params": {}}
        if summary["type_errors"] > 0:
            return {"action_type": "normalize_values", "column": "event_ts", "params": {}}
        return {"action_type": "normalize_values", "column": "event_user_id", "params": {}}

    if task_id == "transactions":
        if summary["type_errors"] > 0:
            return {"action_type": "normalize_values", "column": "amount_usd", "params": {"strategy": "recalculate_from_base_fx"}}
        if summary["duplicates"] > 0:
            return {"action_type": "drop_duplicates", "params": {"key_cols": ["idempotency_key"], "keep": "first"}}
        if summary["format_violations"] > 0:
            return {"action_type": "normalize_values", "column": "merchant_name", "params": {}}
        return {"action_type": "normalize_values", "column": "transaction_ts", "params": {"strategy": "fill_15m_gaps"}}

    return {"action_type": "submit", "params": {}}


def run_task(task_id: str) -> float:
    obs = call_env("reset", data={"task_id": task_id})
    done = False
    final_score = float(obs["current_score"])
    for _ in range(obs["max_steps"]):
        if done:
            break
        if obs["current_score"] >= 0.99:
            break
        action = pick_action(task_id, obs)
        result = call_env("step", data=action, params={"task_id": task_id})
        obs = result["observation"]
        done = bool(result["done"])
        final_score = float(result["info"].get("final_score", obs["current_score"]))

    if not done:
        result = call_env("step", data={"action_type": "submit", "params": {}}, params={"task_id": task_id})
        final_score = float(result["info"].get("final_score", obs["current_score"]))

    return final_score


if __name__ == "__main__":
    scores = {task: run_task(task) for task in TASKS}
    print(json.dumps({"scores": scores, "mean": round(sum(scores.values()) / len(scores), 4)}, indent=2))
