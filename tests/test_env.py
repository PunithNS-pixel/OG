from __future__ import annotations

from fastapi.testclient import TestClient

from env.server import app


client = TestClient(app)


def test_health() -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"


def test_reset_and_step_orders() -> None:
    reset = client.post("/reset", json={"task_id": "orders"})
    assert reset.status_code == 200
    obs = reset.json()
    assert obs["task_id"] == "orders"

    step = client.post(
        "/step",
        params={"task_id": "orders"},
        json={"action_type": "run_validation", "params": {}},
    )
    assert step.status_code == 200
    body = step.json()
    assert "observation" in body
    assert "reward" in body
    assert body["done"] in (True, False)
