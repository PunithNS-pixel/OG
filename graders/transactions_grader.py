from __future__ import annotations

import pandas as pd

from env.models import ValidationCheck
from graders.common import score_from_checks


def _max_gap_days(series: pd.Series) -> int:
    if series.empty:
        return 999
    ts = pd.to_datetime(series, errors="coerce", utc=True).dropna().sort_values().dt.tz_convert(None)
    if len(ts) < 2:
        return 999
    return int(ts.diff().dt.days.max())


def grade_transactions(df: pd.DataFrame) -> tuple[list[ValidationCheck], float]:
    checks: list[ValidationCheck] = []

    if {"amount_usd", "base_amount", "fx_rate", "is_corrupt"}.issubset(df.columns):
        expected = (pd.to_numeric(df["base_amount"], errors="coerce") * pd.to_numeric(df["fx_rate"], errors="coerce")).round(2)
        actual = pd.to_numeric(df["amount_usd"], errors="coerce").round(2)
        corrupt_mask = df["is_corrupt"].astype(str).isin(["1", "True", "true"]) if df["is_corrupt"].dtype == object else df["is_corrupt"] == 1
        bad_recalc = int((actual[corrupt_mask] - expected[corrupt_mask]).abs().fillna(9999.0).gt(0.01).sum())
    else:
        bad_recalc = len(df)

    checks.append(
        ValidationCheck(
            name="amount_recalculation_accuracy",
            passing=bad_recalc == 0,
            detail=f"corrupt rows with wrong recalculation={bad_recalc}",
            weight=1 / 3,
        )
    )

    if "idempotency_key" in df.columns:
        structural_dupes = int(df.duplicated(subset=["idempotency_key"], keep=False).sum())
    else:
        structural_dupes = len(df)

    checks.append(
        ValidationCheck(
            name="zero_structural_duplicates",
            passing=structural_dupes == 0,
            detail=f"rows with duplicate idempotency_key={structural_dupes}",
            weight=1 / 3,
        )
    )

    if "merchant_name" in df.columns:
        merchant = df["merchant_name"].fillna("").astype(str)
        bad_encoding = int(merchant.str.contains("Ã|Â", regex=True).sum())
    else:
        bad_encoding = len(df)

    max_gap = _max_gap_days(df.get("transaction_ts", pd.Series(dtype=str)))
    time_ok = max_gap <= 1
    checks.append(
        ValidationCheck(
            name="encoding_and_time_series",
            passing=(bad_encoding == 0 and time_ok),
            detail=f"encoding anomalies={bad_encoding}, max timestamp gap days={max_gap}",
            weight=1 / 3,
        )
    )

    return checks, score_from_checks(checks)
