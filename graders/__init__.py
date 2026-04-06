from __future__ import annotations

import pandas as pd

from env.models import ValidationCheck
from graders.orders_grader import grade_orders
from graders.transactions_grader import grade_transactions
from graders.users_grader import grade_users


def run_grader(task_id: str, df: pd.DataFrame) -> tuple[list[ValidationCheck], float]:
    if task_id == "orders":
        return grade_orders(df)
    if task_id == "user-merge":
        return grade_users(df)
    if task_id == "transactions":
        return grade_transactions(df)
    raise ValueError(f"No grader registered for task_id={task_id}")
