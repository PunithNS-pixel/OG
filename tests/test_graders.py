from __future__ import annotations

from data.generate import generate_orders, generate_transactions, generate_users
from graders import run_grader


def test_orders_grader_returns_score() -> None:
    checks, score = run_grader("orders", generate_orders())
    assert len(checks) == 6
    assert 0.0 <= score <= 1.0


def test_users_grader_returns_score() -> None:
    checks, score = run_grader("user-merge", generate_users())
    assert len(checks) == 5
    assert 0.0 <= score <= 1.0


def test_transactions_grader_returns_score() -> None:
    checks, score = run_grader("transactions", generate_transactions())
    assert len(checks) == 3
    assert 0.0 <= score <= 1.0
