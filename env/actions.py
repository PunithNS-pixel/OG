from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from env.models import Action


@dataclass
class ActionResult:
    df: pd.DataFrame
    message: str


def _normalize_phone(val: Any) -> str:
    if pd.isna(val):
        return "+1-000-0000"
    digits = re.sub(r"\D", "", str(val))
    if len(digits) == 7:
        return f"+1-{digits[:3]}-{digits[3:]}"
    if len(digits) == 10 and digits.startswith("1"):
        return f"+1-{digits[1:4]}-{digits[4:8]}"
    if len(digits) >= 8:
        return f"+1-{digits[-7:-4]}-{digits[-4:]}"
    return "+1-000-0000"


def _normalize_merchant(val: Any) -> Any:
    if pd.isna(val):
        return val
    text = str(val)
    return (
        text.replace("CafÃ©", "Cafe")
        .replace("NiÃ±o", "Nino")
        .replace("MÃ¼ller", "Muller")
        .replace("SÃ£o", "Sao")
    )


def _coerce_order_date_iso(series: pd.Series) -> pd.Series:
    iso = pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")
    us = pd.to_datetime(series, format="%m/%d/%y", errors="coerce")
    parsed = iso.fillna(us)
    return parsed.dt.strftime("%Y-%m-%d")


def apply_action(df: pd.DataFrame, task_id: str, action: Action) -> ActionResult:
    data = df.copy()
    at = action.action_type
    col = action.column
    params = action.params or {}

    if at.startswith("detect_") or at == "profile_column":
        return ActionResult(df=data, message=f"Detection action '{at}' executed")

    if at == "fill_nulls":
        if col is None:
            raise ValueError("fill_nulls requires a target column")
        strategy = params.get("strategy", "value")
        if strategy == "mean":
            value = pd.to_numeric(data[col], errors="coerce").mean()
        elif strategy == "median":
            value = pd.to_numeric(data[col], errors="coerce").median()
        elif strategy == "mode":
            mode = data[col].dropna().mode()
            value = mode.iloc[0] if not mode.empty else ""
        elif strategy == "value":
            value = params.get("value", "")
        else:
            raise ValueError(f"Unsupported fill strategy '{strategy}'")
        data[col] = data[col].fillna(value)
        return ActionResult(df=data, message=f"Filled nulls in {col} using {strategy}")

    if at == "drop_duplicates":
        key_cols = params.get("key_cols")
        keep = params.get("keep", "first")
        before = len(data)
        data = data.drop_duplicates(subset=key_cols, keep=keep).reset_index(drop=True)
        return ActionResult(df=data, message=f"Dropped {before - len(data)} duplicate rows")

    if at == "cast_column":
        if col is None:
            raise ValueError("cast_column requires a target column")
        target = params.get("target_type")
        if target is None:
            raise ValueError("cast_column requires params.target_type")
        if target in {"float", "int"}:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            if target == "int":
                data[col] = data[col].round().astype("Int64")
        elif target == "datetime":
            if col == "order_date":
                data[col] = _coerce_order_date_iso(data[col])
            else:
                dt = pd.to_datetime(data[col], errors="coerce", utc=True)
                data[col] = dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif target == "str":
            data[col] = data[col].astype(str)
        else:
            raise ValueError(f"Unsupported cast target '{target}'")
        return ActionResult(df=data, message=f"Cast column {col} to {target}")

    if at == "normalize_values":
        if col is None:
            raise ValueError("normalize_values requires a target column")
        mapping = params.get("mapping")
        strategy = params.get("strategy")
        if mapping:
            data[col] = data[col].replace(mapping)
        elif col == "amount_usd" and strategy == "recalculate_from_base_fx":
            if {"base_amount", "fx_rate"}.issubset(data.columns):
                data[col] = (pd.to_numeric(data["base_amount"], errors="coerce") * pd.to_numeric(data["fx_rate"], errors="coerce")).round(2)
        elif col == "phone":
            data[col] = data[col].map(_normalize_phone)
        elif col == "event_ts":
            dt = pd.to_datetime(data[col], errors="coerce", utc=True)
            data[col] = dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif col == "event_user_id":
            if "user_id" in data.columns:
                data[col] = data["user_id"]
        elif col == "transaction_ts" and strategy == "fill_15m_gaps":
            dt = pd.to_datetime(data[col], errors="coerce", utc=True).sort_values().dropna()
            if not dt.empty:
                full = pd.date_range(dt.min(), dt.max(), freq="15min", tz="UTC")
                existing = set(dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
                missing = [ts.strftime("%Y-%m-%dT%H:%M:%SZ") for ts in full if ts.strftime("%Y-%m-%dT%H:%M:%SZ") not in existing]
                if missing:
                    template = data.iloc[0].to_dict()
                    fillers = []
                    for ts in missing:
                        row = template.copy()
                        row["transaction_id"] = f"SYNTH-{ts}"
                        row["transaction_ts"] = ts
                        row["idempotency_key"] = f"SYNTH-{ts}"
                        row["is_corrupt"] = 0
                        fillers.append(row)
                    data = pd.concat([data, pd.DataFrame(fillers)], ignore_index=True)
        elif col == "merchant_name":
            data[col] = data[col].map(_normalize_merchant)
        elif col == "email":
            data[col] = data[col].astype(str).str.lower().str.strip()
        elif col == "name":
            data[col] = data[col].astype(str).str.title().str.strip()
        return ActionResult(df=data, message=f"Normalized values in {col}")

    if at == "clip_outliers":
        if col is None:
            raise ValueError("clip_outliers requires a target column")
        method = params.get("method", "iqr")
        series = pd.to_numeric(data[col], errors="coerce")
        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        else:
            bounds = params.get("bounds", [series.min(), series.max()])
            low, high = bounds[0], bounds[1]
        data[col] = series.clip(lower=low, upper=high)
        return ActionResult(df=data, message=f"Clipped outliers in {col} using {method}")

    if at == "drop_rows":
        condition = params.get("condition")
        if not condition:
            raise ValueError("drop_rows requires params.condition")
        keep_mask = ~data.eval(condition)
        dropped = int((~keep_mask).sum())
        data = data.loc[keep_mask].reset_index(drop=True)
        return ActionResult(df=data, message=f"Dropped {dropped} rows by condition")

    if at == "rename_column":
        if col is None:
            raise ValueError("rename_column requires current column in 'column'")
        new_name = params.get("new_name")
        if not new_name:
            raise ValueError("rename_column requires params.new_name")
        data = data.rename(columns={col: new_name})
        return ActionResult(df=data, message=f"Renamed {col} to {new_name}")

    if at == "run_validation":
        return ActionResult(df=data, message="Validation requested")

    if at == "submit":
        return ActionResult(df=data, message="Submission requested")

    raise ValueError(f"Unknown action_type '{at}'")
