from __future__ import annotations

import pandas as pd
from .schema import AnswerFormat

def df_to_table(df: pd.DataFrame) -> dict:
    return {
        "columns": [str(c) for c in df.columns],
        "rows": df.astype(object).where(df.notna(), None).values.tolist(),
    }

def choose_output_format(plan_format: AnswerFormat, df: pd.DataFrame) -> AnswerFormat:
    if df is None or df.empty:
        return "text"
    if plan_format == "chart":
        return "chart"
    return "table"
