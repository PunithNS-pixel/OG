from __future__ import annotations

from fastapi import FastAPI, HTTPException

from env.environment import DataCleaningEnv
from env.models import Action
from env.tasks import TASKS

app = FastAPI(title="Data Cleaning Pipeline - OpenEnv", version="1.0.0")

_ENVS: dict[str, DataCleaningEnv] = {}


def _get_env(task_id: str) -> DataCleaningEnv:
    if task_id not in TASKS:
        valid = ", ".join(TASKS.keys())
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}', valid: {valid}")
    if task_id not in _ENVS:
        _ENVS[task_id] = DataCleaningEnv(task_id=task_id)
    return _ENVS[task_id]


@app.get("/")
def health() -> dict:
    return {"status": "ok", "service": "data-cleaning-openenv", "version": "1.0.0"}


@app.post("/reset")
def reset(body: dict) -> dict:
    task_id = body.get("task_id", "orders")
    env = _get_env(task_id)
    return env.reset().model_dump()


@app.post("/step")
def step(action: Action, task_id: str | None = None) -> dict:
    current_task = task_id if task_id is not None else action.params.get("task_id", "orders")
    env = _get_env(current_task)
    try:
        obs, reward, done, info = env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(task_id: str = "orders") -> dict:
    env = _get_env(task_id)
    return env.state()


@app.get("/tasks")
def tasks() -> dict:
    return {
        "tasks": [
            {
                "id": cfg.task_id,
                "name": cfg.name,
                "difficulty": cfg.difficulty,
                "max_steps": cfg.max_steps,
            }
            for cfg in TASKS.values()
        ]
    }
