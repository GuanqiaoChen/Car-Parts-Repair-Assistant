from __future__ import annotations

import os
import pandas as pd

# Columns we expect to find in the raw CSV backing the assistant.
REQUIRED_COLS = [
    "DemandDate","BuildDate","Province","CountryCode","Quantity","SSPart",
    "RequestType","DemandType","PayType","VehicleModel","VIN"
]


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the raw repair history CSV and perform minimal, repeatable
    preprocessing so downstream plans can assume a consistent shape.

    Responsibilities:
    - enforce the presence of required columns,
    - normalise dates into proper `datetime` objects, and
    - derive `VehicleAgeDays`, which many analyses lean on.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"DATA_PATH not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Date normalisation: coercing bad values to NaT keeps later filters simple.
    df["DemandDate"] = pd.to_datetime(df["DemandDate"], errors="coerce")
    df["BuildDate"] = pd.to_datetime(df["BuildDate"], errors="coerce")

    # Feature engineering: vehicle age at demand time (days).
    # This is computed once at load time to keep query‑time plans cheap.
    df["VehicleAgeDays"] = (df["DemandDate"] - df["BuildDate"]).dt.days

    # Basic missing handling: we keep rows and let downstream filters/
    # aggregations decide how to treat NaNs for a given analysis.
    return df
