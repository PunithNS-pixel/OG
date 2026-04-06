from __future__ import annotations

import re

import pandas as pd

from env.models import ValidationCheck
from graders.common import score_from_checks

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
VALID_STATUS = {"shipped", "pending", "delivered", "cancelled"}


def grade_orders(df: pd.DataFrame) -> tuple[list[ValidationCheck], float]:
    checks: list[ValidationCheck] = []

    nulls = int(pd.to_numeric(df.get("amount_usd"), errors="coerce").isna().sum())
    checks.append(
        ValidationCheck(
            name="amount_usd_no_nulls",
            passing=nulls == 0,
            detail=f"null amount_usd rows={nulls}",
            weight=1 / 6,
        )
    )

    dupes = int(df.duplicated(subset=["order_id"], keep=False).sum()) if "order_id" in df.columns else len(df)
    checks.append(
        ValidationCheck(
            name="order_id_unique",
            passing=dupes == 0,
            detail=f"duplicate order_id rows={dupes}",
            weight=1 / 6,
        )
    )

    emails = df.get("email", pd.Series(dtype=str)).fillna("").astype(str)
    invalid_emails = int((~emails.map(lambda x: bool(EMAIL_RE.match(x)))).sum())
    checks.append(
        ValidationCheck(
            name="email_regex",
            passing=invalid_emails == 0,
            detail=f"invalid emails={invalid_emails}",
            weight=1 / 6,
        )
    )

    dates = df.get("order_date", pd.Series(dtype=str)).fillna("").astype(str)
    parseable = int(dates.map(lambda x: bool(DATE_RE.match(x))).sum())
    bad_dates = int(len(dates) - parseable)
    checks.append(
        ValidationCheck(
            name="date_format_iso",
            passing=bad_dates == 0,
            detail=f"non ISO dates={bad_dates}",
            weight=1 / 6,
        )
    )

    statuses = set(df.get("status", pd.Series(dtype=str)).dropna().astype(str).unique())
    bad_statuses = sorted(s for s in statuses if s not in VALID_STATUS)
    checks.append(
        ValidationCheck(
            name="status_domain",
            passing=len(bad_statuses) == 0,
            detail=f"invalid statuses={bad_statuses}",
            weight=1 / 6,
        )
    )

    amount = pd.to_numeric(df.get("amount_usd"), errors="coerce")
    out_of_range = int(((amount < 0) | (amount > 10000) | amount.isna()).sum())
    checks.append(
        ValidationCheck(
            name="amount_range",
            passing=out_of_range == 0,
            detail=f"amounts outside [0,10000]={out_of_range}",
            weight=1 / 6,
        )
    )

    return checks, score_from_checks(checks)
