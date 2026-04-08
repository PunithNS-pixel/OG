from __future__ import annotations

import json
import os
import sys
import time
from urllib import error, parse, request

ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")
TASKS = ["orders", "user-merge", "transactions"]


class EnvClientError(RuntimeError):
    """Raised when the benchmark env cannot be reached or returns invalid data."""


def call_env(endpoint: str, method: str = "POST", data: dict | None = None, params: dict | None = None) -> dict:
    url = f"{ENV_URL}/{endpoint}" if endpoint else ENV_URL
    if params:
        query = parse.urlencode(params)
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}{query}"

    body: bytes | None = None
    headers = {"Accept": "application/json"}
    if method == "POST":
        body = json.dumps(data or {}).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url=url, data=body, method=method, headers=headers)
    try:
        with request.urlopen(req, timeout=45) as response:
            raw = response.read().decode("utf-8")
    except (error.HTTPError, error.URLError, TimeoutError, OSError) as exc:
        raise EnvClientError(f"Request failed for {method} {url}: {exc}") from exc

    try:
        return json.loads(raw)
    except ValueError as exc:
        raise EnvClientError(f"Invalid JSON from {method} {url}") from exc


def wait_for_env(max_retries: int = 8, base_delay: float = 1.0) -> bool:
    for attempt in range(1, max_retries + 1):
        try:
            call_env("", method="GET")
            return True
        except EnvClientError as exc:
            if attempt == max_retries:
                print(f"[warn] env unreachable at {ENV_URL}: {exc}", file=sys.stderr)
                return False
            time.sleep(base_delay * attempt)
    return False


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
    try:
        obs = call_env("reset", data={"task_id": task_id})
        done = False
        final_score = float(obs.get("current_score", 0.0))

        for _ in range(int(obs.get("max_steps", 0))):
            if done:
                break
            if float(obs.get("current_score", 0.0)) >= 0.99:
                break

            action = pick_action(task_id, obs)
            result = call_env("step", data=action, params={"task_id": task_id})
            obs = result.get("observation", {})
            done = bool(result.get("done", False))
            info = result.get("info", {})
            final_score = float(info.get("final_score", obs.get("current_score", final_score)))

        if not done:
            result = call_env("step", data={"action_type": "submit", "params": {}}, params={"task_id": task_id})
            info = result.get("info", {})
            final_score = float(info.get("final_score", obs.get("current_score", final_score)))

        return final_score
    except (EnvClientError, KeyError, TypeError, ValueError) as exc:
        print(f"[warn] task '{task_id}' failed: {exc}", file=sys.stderr)
        return 0.0
    except BaseException as exc:
        print(f"[warn] task '{task_id}' failed unexpectedly: {exc}", file=sys.stderr)
        return 0.0


if __name__ == "__main__":
    if not wait_for_env():
        fallback_scores = {task: 0.0 for task in TASKS}
        print(json.dumps({"scores": fallback_scores, "mean": 0.0, "status": "env_unreachable"}, indent=2))
        raise SystemExit(0)

    try:
        scores = {task: run_task(task) for task in TASKS}
        mean_score = round(sum(scores.values()) / len(scores), 4) if scores else 0.0
        print(json.dumps({"scores": scores, "mean": mean_score}, indent=2))
    except BaseException as exc:
        print(f"[warn] inference failed unexpectedly: {exc}", file=sys.stderr)
        fallback_scores = {task: 0.0 for task in TASKS}
        print(json.dumps({"scores": fallback_scores, "mean": 0.0, "status": "inference_error"}, indent=2))
        raise SystemExit(0)