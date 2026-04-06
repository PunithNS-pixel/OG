from __future__ import annotations

from typing import Iterable

from env.models import ValidationCheck


def score_from_checks(checks: Iterable[ValidationCheck]) -> float:
    checks = list(checks)
    total_weight = sum(c.weight for c in checks)
    if total_weight <= 0:
        return 0.0
    passed_weight = sum(c.weight for c in checks if c.passing)
    return round(passed_weight / total_weight, 4)
