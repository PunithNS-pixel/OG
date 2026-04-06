from __future__ import annotations

import re

import pandas as pd

from env.models import ValidationCheck
from graders.common import score_from_checks

PHONE_RE = re.compile(r"^\+1-\d{3}-\d{4}$")


def grade_users(df: pd.DataFrame) -> tuple[list[ValidationCheck], float]:
    checks: list[ValidationCheck] = []

    dedup_count = int(df.duplicated(subset=["email"], keep=False).sum()) if "email" in df.columns else len(df)
    checks.append(
        ValidationCheck(
            name="dedup_accuracy",
            passing=dedup_count == 0,
            detail=f"rows still duplicated by email={dedup_count}",
            weight=0.2,
        )
    )

    phones = df.get("phone", pd.Series(dtype=str)).fillna("").astype(str)
    bad_phone = int((~phones.map(lambda x: bool(PHONE_RE.match(x)))).sum())
    checks.append(
        ValidationCheck(
            name="phone_normalization",
            passing=bad_phone == 0,
            detail=f"phone rows not normalized={bad_phone}",
            weight=0.2,
        )
    )

    ts = df.get("event_ts", pd.Series(dtype=str)).fillna("").astype(str)
    utc = int(ts.str.endswith("Z").sum())
    non_utc = int(len(ts) - utc)
    checks.append(
        ValidationCheck(
            name="utc_conversion",
            passing=non_utc == 0,
            detail=f"timestamps not UTC={non_utc}",
            weight=0.2,
        )
    )

    critical_cols = ["name", "phone", "event_ts", "event_name"]
    missing_cols = [c for c in critical_cols if c not in df.columns]
    nulls = int(df[critical_cols].isna().sum().sum()) if not missing_cols else len(df)
    checks.append(
        ValidationCheck(
            name="null_fill",
            passing=(len(missing_cols) == 0 and nulls == 0),
            detail=f"missing columns={missing_cols}, null cells in critical columns={nulls}",
            weight=0.2,
        )
    )

    if {"user_id", "event_user_id"}.issubset(df.columns):
        ri_breaks = int((df["user_id"].astype(str) != df["event_user_id"].astype(str)).sum())
    else:
        ri_breaks = len(df)
    checks.append(
        ValidationCheck(
            name="referential_integrity",
            passing=ri_breaks == 0,
            detail=f"user_id/event_user_id mismatches={ri_breaks}",
            weight=0.2,
        )
    )

    return checks, score_from_checks(checks)
