from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42


def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_orders(seed: int = SEED) -> pd.DataFrame:
    rng = _rng(seed)
    n_rows = 490

    order_ids = [f"ORD-{i:06d}" for i in range(1, n_rows + 1)]
    emails = [f"customer{i}@example.com" for i in range(1, n_rows + 1)]
    dates = pd.date_range("2025-01-01", periods=n_rows, freq="D").strftime("%Y-%m-%d").tolist()
    statuses = rng.choice(["shipped", "pending", "delivered", "cancelled"], size=n_rows).tolist()
    amounts = rng.uniform(5, 450, size=n_rows).round(2)

    df = pd.DataFrame(
        {
            "order_id": order_ids,
            "email": emails,
            "order_date": dates,
            "status": statuses,
            "amount_usd": amounts,
        }
    )

    dup_source = rng.choice(df.index, size=10, replace=False)
    df = pd.concat([df, df.loc[dup_source]], ignore_index=True)

    bad_amount_idx = rng.choice(df.index, size=15, replace=False)
    df.loc[bad_amount_idx, "amount_usd"] = np.nan

    bad_date_idx = rng.choice(df.index.difference(bad_amount_idx), size=20, replace=False)
    dt = pd.to_datetime(df.loc[bad_date_idx, "order_date"], errors="coerce")
    df.loc[bad_date_idx, "order_date"] = dt.dt.strftime("%m/%d/%y")

    bad_email_idx = rng.choice(df.index.difference(np.union1d(bad_amount_idx, bad_date_idx)), size=4, replace=False)
    df.loc[bad_email_idx, "email"] = ["badmail", "bad@", "@bad.com", "a b@x.com"]

    bad_status_idx = rng.choice(df.index.difference(np.union1d(np.union1d(bad_amount_idx, bad_date_idx), bad_email_idx)), size=12, replace=False)
    df.loc[bad_status_idx, "status"] = df.loc[bad_status_idx, "status"].str.upper()

    outlier_idx = rng.choice(df.index.difference(bad_amount_idx), size=3, replace=False)
    df.loc[outlier_idx, "amount_usd"] = [12000.0, 15000.0, 18000.0]

    neg_pool = df.index.difference(np.union1d(outlier_idx, bad_amount_idx))
    neg_idx = rng.choice(neg_pool, size=5, replace=False)
    df.loc[neg_idx, "amount_usd"] = [-10.0, -18.0, -25.0, -8.0, -99.0]

    return df.reset_index(drop=True)


def generate_users(seed: int = SEED) -> pd.DataFrame:
    rng = _rng(seed + 1)
    n_rows = 1200

    user_ids = np.arange(10000, 10000 + n_rows)
    names = [f"User {i}" for i in user_ids]
    emails = [f"user{i}@company.com" for i in user_ids]
    phones = [f"+1-{rng.integers(200, 999)}-{rng.integers(1000, 9999)}" for _ in user_ids]
    timestamps = pd.date_range("2025-02-01", periods=n_rows, freq="h", tz="UTC").astype(str).str.replace("+00:00", "Z")

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "name": names,
            "email": emails,
            "phone": phones,
            "event_user_id": user_ids,
            "event_name": rng.choice(["login", "purchase", "logout", "view"], size=n_rows),
            "event_ts": timestamps,
        }
    )

    near_dupe_idx = rng.choice(df.index, size=35, replace=False)
    near_dupe_rows = df.loc[near_dupe_idx].copy()
    near_dupe_rows["name"] = near_dupe_rows["name"].str.swapcase()
    near_dupe_rows["email"] = near_dupe_rows["email"].str.upper()
    df.loc[near_dupe_idx, "email"] = near_dupe_rows["email"].values

    mixed_phone_idx = rng.choice(df.index, size=220, replace=False)
    df.loc[mixed_phone_idx, "phone"] = df.loc[mixed_phone_idx, "phone"].str.replace("+1-", "", regex=False)

    local_ts_idx = rng.choice(df.index, size=260, replace=False)
    ts_local = pd.to_datetime(df.loc[local_ts_idx, "event_ts"], utc=True).dt.tz_convert("America/New_York")
    df.loc[local_ts_idx, "event_ts"] = ts_local.dt.strftime("%Y-%m-%d %H:%M:%S")

    wrong_join_rows = rng.choice(df.index, size=80, replace=False)
    wrong_join_cols = ["name", "phone", "event_name", "event_ts"]
    df.loc[wrong_join_rows, wrong_join_cols] = np.nan

    ri_break_rows = rng.choice(df.index.difference(wrong_join_rows), size=95, replace=False)
    df.loc[ri_break_rows, "event_user_id"] = df.loc[ri_break_rows, "event_user_id"] + 999999

    return df.reset_index(drop=True)


def generate_transactions(seed: int = SEED) -> pd.DataFrame:
    rng = _rng(seed + 2)
    n_rows = 2000

    tx_ids = [f"TX-{i:07d}" for i in range(1, n_rows + 1)]
    base_amount = rng.uniform(10, 500, size=n_rows).round(2)
    fx_rate = rng.choice([1.0, 1.1, 0.9, 1.25], size=n_rows)
    amount_usd = (base_amount * fx_rate).round(2)

    timestamps = pd.date_range("2025-01-01", periods=n_rows, freq="15min", tz="UTC")
    gap_start = 850
    gap_len = 3 * 24 * 4
    timestamps = timestamps.delete(np.arange(gap_start, gap_start + gap_len))
    while len(timestamps) < n_rows:
        timestamps = timestamps.append(pd.DatetimeIndex([timestamps[-1] + pd.Timedelta(minutes=15)]))
    timestamps = timestamps[:n_rows]

    merchants = rng.choice(["Cafe Central", "Nino Store", "Muller GmbH", "Sao Mart"], size=n_rows).astype(object)
    bad_encoding_idx = rng.choice(np.arange(n_rows), size=120, replace=False)
    merchants[bad_encoding_idx[:30]] = "CafÃ© Central"
    merchants[bad_encoding_idx[30:60]] = "NiÃ±o Store"
    merchants[bad_encoding_idx[60:90]] = "MÃ¼ller GmbH"
    merchants[bad_encoding_idx[90:]] = "SÃ£o Mart"

    idempotency_keys = [f"IDEMP-{i:06d}" for i in range(1, n_rows + 1)]
    structural_dup_idx = rng.choice(np.arange(n_rows), size=140, replace=False)
    dup_targets = rng.choice(np.setdiff1d(np.arange(n_rows), structural_dup_idx), size=140, replace=False)
    for src, tgt in zip(structural_dup_idx, dup_targets, strict=False):
        idempotency_keys[src] = idempotency_keys[tgt]

    df = pd.DataFrame(
        {
            "transaction_id": tx_ids,
            "base_amount": base_amount,
            "fx_rate": fx_rate,
            "amount_usd": amount_usd,
            "merchant_name": merchants,
            "idempotency_key": idempotency_keys,
            "transaction_ts": pd.DatetimeIndex(timestamps).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "is_corrupt": 0,
        }
    )

    corrupt_idx = rng.choice(df.index, size=340, replace=False)
    corruption_factor = rng.choice([1.5, 2.0, 2.5], size=340)
    df.loc[corrupt_idx, "amount_usd"] = (df.loc[corrupt_idx, "amount_usd"].to_numpy() * corruption_factor).round(2)
    df.loc[corrupt_idx, "is_corrupt"] = 1

    return df.reset_index(drop=True)


def save_datasets(out_dir: str | Path = Path(__file__).resolve().parent) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    generate_orders().to_csv(out / "orders_dirty.csv", index=False)
    generate_users().to_csv(out / "users_dirty.csv", index=False)
    generate_transactions().to_csv(out / "transactions_dirty.csv", index=False)


if __name__ == "__main__":
    save_datasets()
    print("Generated deterministic datasets in data/")
