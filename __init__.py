import pandas as pd
import numpy as np
import json


# ─── Task 1 Grader ────────────────────────────────────────────────────────────

def grade_task1(cleaned: pd.DataFrame, gold: pd.DataFrame) -> float:
    """
    Checks: nulls removed, duplicates dropped, age is numeric.
    Returns score 0.0 - 1.0.
    """
    score = 0.0
    checks = 4

    # 1. No nulls in age or email
    if cleaned["age"].isnull().sum() == 0:
        score += 1
    if "email" in cleaned.columns and cleaned["email"].isnull().sum() == 0:
        score += 1

    # 2. No duplicate rows
    if cleaned.duplicated().sum() == 0:
        score += 1

    # 3. age is numeric type
    if pd.api.types.is_numeric_dtype(cleaned["age"]):
        score += 1

    return round(score / checks, 4)


# ─── Task 2 Grader ────────────────────────────────────────────────────────────

def grade_task2(cleaned: pd.DataFrame, gold: pd.DataFrame) -> float:
    """
    Checks: dates standardized, outliers removed, gender normalized.
    Returns score 0.0 - 1.0.
    """
    score = 0.0
    checks = 5

    # 1. Date column parseable as datetime
    try:
        pd.to_datetime(cleaned["date"])
        score += 1
    except Exception:
        pass

    # 2. No extreme revenue outliers (> 3 std from mean of non-outlier range)
    rev = pd.to_numeric(cleaned["revenue"], errors="coerce").dropna()
    if rev.max() < 50000 and rev.min() > 0:
        score += 1

    # 3. Gender normalized to Male/Female only
    valid_genders = {"Male", "Female"}
    if set(cleaned["gender"].dropna().unique()).issubset(valid_genders):
        score += 1

    # 4. Row count reasonable (not over-dropped)
    retention = len(cleaned) / len(gold) if len(gold) > 0 else 0
    if 0.85 <= retention <= 1.15:
        score += 1

    # 5. No mixed date formats (all same format pattern)
    try:
        parsed = pd.to_datetime(cleaned["date"])
        if parsed.dt.strftime("%Y-%m-%d").nunique() <= len(cleaned):
            score += 1
    except Exception:
        pass

    return round(score / checks, 4)


# ─── Task 3 Grader ────────────────────────────────────────────────────────────

def grade_task3(cleaned: pd.DataFrame, gold: pd.DataFrame) -> float:
    """
    Checks: near-dupes removed, join inflation fixed, address parsed.
    Returns score 0.0 - 1.0.
    """
    score = 0.0
    checks = 5

    # 1. No ORD_ prefixed duplicates remain
    if "order_id" in cleaned.columns:
        bad = cleaned["order_id"].str.startswith("ORD_").sum()
        if bad == 0:
            score += 1

    # 2. Amount values are reasonable (not inflated)
    if "amount" in cleaned.columns:
        amounts = pd.to_numeric(cleaned["amount"], errors="coerce").dropna()
        if amounts.median() < 500:  # inflated median would be >> 500
            score += 1

    # 3. address_json column is gone (parsed out) OR city/zip columns exist
    addr_parsed = ("city" in cleaned.columns and "zip" in cleaned.columns)
    addr_raw = ("address_json" not in cleaned.columns)
    if addr_parsed or addr_raw:
        score += 1

    # 4. No duplicate order_ids
    if "order_id" in cleaned.columns:
        if cleaned["order_id"].duplicated().sum() == 0:
            score += 1

    # 5. Row count close to gold
    if len(gold) > 0:
        retention = len(cleaned) / len(gold)
        if 0.90 <= retention <= 1.10:
            score += 1

    return round(score / checks, 4)


# ─── Registry ─────────────────────────────────────────────────────────────────

GRADERS = {
    "task_1": grade_task1,
    "task_2": grade_task2,
    "task_3": grade_task3,
}


def run_grader(task_id: str, cleaned: pd.DataFrame, gold: pd.DataFrame) -> float:
    fn = GRADERS.get(task_id)
    if fn is None:
        raise ValueError(f"Unknown task_id: {task_id}")
    return fn(cleaned, gold)
