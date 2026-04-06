from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    name: str
    difficulty: str
    dataset_path: Path
    max_steps: int


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

TASKS: dict[str, TaskConfig] = {
    "orders": TaskConfig(
        task_id="orders",
        name="E-commerce order cleanup",
        difficulty="easy",
        dataset_path=DATA_DIR / "orders_dirty.csv",
        max_steps=15,
    ),
    "user-merge": TaskConfig(
        task_id="user-merge",
        name="Multi-source user profile merge",
        difficulty="medium",
        dataset_path=DATA_DIR / "users_dirty.csv",
        max_steps=22,
    ),
    "transactions": TaskConfig(
        task_id="transactions",
        name="Financial transaction audit trail",
        difficulty="hard",
        dataset_path=DATA_DIR / "transactions_dirty.csv",
        max_steps=30,
    ),
}


def get_task(task_id: str) -> TaskConfig:
    if task_id not in TASKS:
        valid = ", ".join(TASKS.keys())
        raise ValueError(f"Unknown task_id '{task_id}'. Expected one of: {valid}")
    return TASKS[task_id]
