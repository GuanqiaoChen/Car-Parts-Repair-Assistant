from __future__ import annotations

import pandas as pd
from .schema import AnswerFormat


def df_to_table(df: pd.DataFrame) -> dict:
    """
    Convert an analysis DataFrame into the lightweight table structure the
    frontend understands ('columns' + 'rows', with NaNs normalised to None).
    """
    return {
        "columns": [str(c) for c in df.columns],
        "rows": df.astype(object).where(df.notna(), None).values.tolist(),
    }


def choose_output_format(plan_format: AnswerFormat, df: pd.DataFrame) -> AnswerFormat:
    """
    Decide whether a given step should render as text, table, or chart.

    The planner can express a preference, but we still downgrade to plain text
    for empty results and fall back to a table when no chart makes sense.
    """
    if df is None or df.empty:
        return "text"
    if plan_format == "chart":
        return "chart"
    return "table"
