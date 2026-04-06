from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ActionType = Literal[
    "detect_nulls",
    "detect_duplicates",
    "detect_type_errors",
    "detect_format_violations",
    "detect_outliers",
    "profile_column",
    "fill_nulls",
    "drop_duplicates",
    "cast_column",
    "normalize_values",
    "clip_outliers",
    "drop_rows",
    "rename_column",
    "run_validation",
    "submit",
]


class Action(BaseModel):
    action_type: ActionType
    column: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class ValidationCheck(BaseModel):
    name: str
    passing: bool
    detail: str
    weight: float


class Observation(BaseModel):
    task_id: str
    step: int
    max_steps: int
    shape: tuple[int, int]
    dtypes: dict[str, str]
    null_counts: dict[str, int]
    duplicate_count: int
    error_summary: dict[str, int]
    sample_rows: list[dict[str, Any]]
    validation_report: list[ValidationCheck]
    current_score: float
