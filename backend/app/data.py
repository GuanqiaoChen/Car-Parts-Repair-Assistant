from __future__ import annotations

import os
import pandas as pd

REQUIRED_COLS = [
    "DemandDate","BuildDate","Province","CountryCode","Quantity","SSPart",
    "RequestType","DemandType","PayType","VehicleModel","VIN"
]

def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"DATA_PATH not found: {path}")

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Date normalization
    df["DemandDate"] = pd.to_datetime(df["DemandDate"], errors="coerce")
    df["BuildDate"] = pd.to_datetime(df["BuildDate"], errors="coerce")

    # Feature engineering: vehicle age at demand time (days)
    df["VehicleAgeDays"] = (df["DemandDate"] - df["BuildDate"]).dt.days

    # Basic missing handling: keep rows; downstream filters/aggregations should handle NaNs
    return df
