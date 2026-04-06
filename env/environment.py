from __future__ import annotations

from pathlib import Path

import pandas as pd

from env.actions import apply_action
from env.models import Action, Observation, ValidationCheck
from env.tasks import get_task
from graders import run_grader


class DataCleaningEnv:
    def __init__(self, task_id: str):
        self.task = get_task(task_id)
        self.task_id = task_id
        self.max_steps = self.task.max_steps

        self._df_original: pd.DataFrame | None = None
        self._df: pd.DataFrame | None = None
        self._step = 0
        self._done = False
        self._last_report: list[ValidationCheck] = []
        self._last_score = 0.0

        self._load_dataset()

    def _load_dataset(self) -> None:
        dataset_path = Path(self.task.dataset_path)
        if not dataset_path.exists():
            from data.generate import save_datasets

            save_datasets(dataset_path.parent)
        self._df_original = pd.read_csv(dataset_path)

    def reset(self) -> Observation:
        self._df = self._df_original.copy(deep=True)
        self._step = 0
        self._done = False
        self._last_report, self._last_score = run_grader(self.task_id, self._df)
        return self._make_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset first.")

        old_score = self._last_score
        result = apply_action(self._df, self.task_id, action)
        self._df = result.df

        self._step += 1
        report, score = run_grader(self.task_id, self._df)
        self._last_report = report
        self._last_score = score

        reward = round(score - old_score, 4)
        info = {
            "message": result.message,
            "score_before": old_score,
            "score_after": score,
        }

        if action.action_type == "submit" or self._step >= self.max_steps:
            self._done = True
            info["final_score"] = score

        return self._make_observation(), reward, self._done, info

    def state(self) -> dict:
        return {
            "task_id": self.task_id,
            "step": self._step,
            "max_steps": self.max_steps,
            "done": self._done,
            "score": self._last_score,
            "shape": list(self._df.shape),
        }

    def _error_summary(self) -> dict[str, int]:
        summary = {"nulls": 0, "duplicates": 0, "type_errors": 0, "format_violations": 0, "outliers": 0}
        df = self._df

        summary["nulls"] = int(df.isna().sum().sum())

        key_map = {
            "orders": ["order_id"],
            "user-merge": ["email"],
            "transactions": ["idempotency_key"],
        }
        key_cols = key_map[self.task_id]
        summary["duplicates"] = int(df.duplicated(subset=key_cols, keep=False).sum())

        if self.task_id == "orders":
            amt = pd.to_numeric(df["amount_usd"], errors="coerce")
            summary["outliers"] = int(((amt > 10000) | (amt < 0)).sum())
            summary["format_violations"] = int((~df["order_date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")).sum())
            summary["type_errors"] = int(amt.isna().sum())
        elif self.task_id == "user-merge":
            summary["format_violations"] = int((~df["phone"].fillna("").astype(str).str.match(r"^\+1-\d{3}-\d{4}$")).sum())
            summary["type_errors"] = int((~df["event_ts"].fillna("").astype(str).str.endswith("Z")).sum())
        else:
            expected = (pd.to_numeric(df["base_amount"], errors="coerce") * pd.to_numeric(df["fx_rate"], errors="coerce")).round(2)
            actual = pd.to_numeric(df["amount_usd"], errors="coerce").round(2)
            corrupt_mask = df["is_corrupt"] == 1
            summary["type_errors"] = int((actual[corrupt_mask] - expected[corrupt_mask]).abs().fillna(9999).gt(0.01).sum())
            summary["format_violations"] = int(df["merchant_name"].fillna("").astype(str).str.contains("Ã|Â", regex=True).sum())

        return summary

    def _make_observation(self) -> Observation:
        return Observation(
            task_id=self.task_id,
            step=self._step,
            max_steps=self.max_steps,
            shape=(int(self._df.shape[0]), int(self._df.shape[1])),
            dtypes={c: str(t) for c, t in self._df.dtypes.items()},
            null_counts={c: int(v) for c, v in self._df.isna().sum().to_dict().items()},
            duplicate_count=self._error_summary()["duplicates"],
            error_summary=self._error_summary(),
            sample_rows=self._df.head(5).fillna("NULL").to_dict(orient="records"),
            validation_report=self._last_report,
            current_score=self._last_score,
        )
