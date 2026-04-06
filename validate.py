from __future__ import annotations

import json

import requests

BASE_URL = "http://localhost:8000"
TASKS = ["orders", "user-merge", "transactions"]


def main() -> None:
    health = requests.get(f"{BASE_URL}/", timeout=20)
    health.raise_for_status()

    results: dict[str, float] = {}
    for task in TASKS:
        obs = requests.post(f"{BASE_URL}/reset", json={"task_id": task}, timeout=20).json()
        max_steps = obs["max_steps"]

        for _ in range(min(3, max_steps)):
            step_result = requests.post(
                f"{BASE_URL}/step",
                params={"task_id": task},
                json={"action_type": "run_validation", "params": {}},
                timeout=20,
            ).json()
            obs = step_result["observation"]

        submit_result = requests.post(
            f"{BASE_URL}/step",
            params={"task_id": task},
            json={"action_type": "submit", "params": {}},
            timeout=20,
        ).json()
        results[task] = float(submit_result["info"].get("final_score", obs["current_score"]))

    print(json.dumps({"validation": "ok", "scores": results}, indent=2))


if __name__ == "__main__":
    main()
