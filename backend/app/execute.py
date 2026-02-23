from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional

from .schema import AssistantPlan, SimpleGroupbyPlan, PivotPlan, TrendPlan, FirstRepairDelayPlan

def _apply_filters(df: pd.DataFrame, filters) -> pd.DataFrame:
    out = df
    for f in filters:
        col, op, val = f.col, f.op, f.value
        if col not in out.columns:
            continue

        if op == "==":
            out = out[out[col] == val]
        elif op == "!=":
            out = out[out[col] != val]
        elif op == "in":
            vals = val if isinstance(val, list) else [val]
            out = out[out[col].isin(vals)]
        elif op == "contains":
            out = out[out[col].astype(str).str.contains(str(val), na=False)]
        elif op == ">=":
            out = out[out[col] >= val]
        elif op == "<=":
            out = out[out[col] <= val]
        elif op == "between":
            if isinstance(val, list) and len(val) == 2:
                lo, hi = val[0], val[1]
                out = out[(out[col] >= lo) & (out[col] <= hi)]
    return out

def _cap_df(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0:
        return df
    return df.head(max_rows)

def run_plan(df: pd.DataFrame, plan: AssistantPlan, max_rows_returned: int = 200) -> Tuple[pd.DataFrame, Optional[str]]:
    a = plan.analysis

    if isinstance(a, SimpleGroupbyPlan):
        dff = _apply_filters(df, a.filters)

        # If no groupby, just aggregate whole dataset
        if not a.groupby:
            out = {}
            if not a.metrics:
                out["row_count"] = int(len(dff))
                return pd.DataFrame([out]), None

            for m in a.metrics:
                if m.col == "row_count":
                    series = pd.Series(np.ones(len(dff), dtype=int))
                else:
                    series = dff[m.col]
                alias = m.alias or f"{m.agg}_{m.col}"
                if m.agg == "count":
                    out[alias] = int(series.count())
                elif m.agg == "sum":
                    out[alias] = float(series.sum())
                elif m.agg == "mean":
                    out[alias] = float(series.mean())
                elif m.agg == "median":
                    out[alias] = float(series.median())
                elif m.agg == "min":
                    out[alias] = float(series.min())
                elif m.agg == "max":
                    out[alias] = float(series.max())
                elif m.agg == "nunique":
                    out[alias] = int(series.nunique())
            return pd.DataFrame([out]), None

        gb = dff.groupby(list(a.groupby), dropna=False)

        if not a.metrics:
            res = gb.size().reset_index(name="row_count")
        else:
            agg_dict = {}
            for m in a.metrics:
                alias = m.alias or f"{m.agg}_{m.col}"
                if m.col == "row_count":
                    # count rows in group
                    agg_dict[alias] = ("VIN", "count") if "VIN" in dff.columns else (dff.columns[0], "count")
                else:
                    agg_dict[alias] = (m.col, m.agg if m.agg != "median" else "median")
            res = gb.agg(**agg_dict).reset_index()

        if a.sort is not None and a.sort.by in res.columns:
            res = res.sort_values(by=a.sort.by, ascending=a.sort.ascending)

        res = res.head(max(1, int(a.limit)))
        return _cap_df(res, max_rows_returned), None

    if isinstance(a, PivotPlan):
        dff = _apply_filters(df, a.filters)
        value_col = "Quantity" if a.value == "Quantity" else None

        if a.agg == "sum" and value_col is None:
            return pd.DataFrame([]), "Pivot sum requires Quantity as value."

        if a.value == "row_count":
            dff["_row_count"] = 1
            value_col = "_row_count"

        piv = pd.pivot_table(
            dff,
            index=a.rows,
            columns=a.cols,
            values=value_col,
            aggfunc=("sum" if a.agg == "sum" else "count"),
            fill_value=0,
        )

        # cap size
        piv = piv.iloc[: a.limit_rows, : a.limit_cols]

        # Flatten for transport
        piv_reset = piv.reset_index()
        return _cap_df(piv_reset, max_rows_returned), None

    if isinstance(a, TrendPlan):
        dff = _apply_filters(df, a.filters)

        if a.date_col not in dff.columns:
            return pd.DataFrame([]), f"Missing date column: {a.date_col}"

        # drop NaT dates
        dff = dff.dropna(subset=[a.date_col]).copy()
        if dff.empty:
            return pd.DataFrame([]), "No rows with valid dates after filtering."

        dff["_bucket"] = dff[a.date_col].dt.to_period(a.freq).dt.to_timestamp()

        if a.metric == "row_count":
            dff["_row_count"] = 1
            val_col = "_row_count"
            agg = "sum"
        else:
            val_col = "Quantity"
            agg = "sum" if a.agg == "sum" else "count"

        if a.groupby is None:
            res = dff.groupby("_bucket", dropna=False)[val_col].agg(agg).reset_index(name="value")
        else:
            # limit to top series by total volume to keep charts readable
            totals = dff.groupby(a.groupby)[val_col].agg(agg).sort_values(ascending=False).head(a.limit_series).index
            dff = dff[dff[a.groupby].isin(totals)]
            res = dff.groupby(["_bucket", a.groupby], dropna=False)[val_col].agg(agg).reset_index(name="value")

        res = res.sort_values("_bucket").head(a.limit_points)
        return _cap_df(res, max_rows_returned), None

    if isinstance(a, FirstRepairDelayPlan):
        dff = _apply_filters(df, a.filters)

        needed = {"VIN", "DemandDate", "BuildDate", a.by}
        missing = [c for c in needed if c not in dff.columns]
        if missing:
            return pd.DataFrame([]), f"Missing required columns for first repair delay: {missing}"

        dff = dff.dropna(subset=["VIN", "DemandDate", "BuildDate"]).copy()
        if dff.empty:
            return pd.DataFrame([]), "No valid rows after filtering for delay computation."

        first = (
            dff.sort_values("DemandDate")
              .groupby("VIN", as_index=False)
              .first()[["VIN", "DemandDate", "BuildDate", a.by]]
        )
        first["delay_days"] = (first["DemandDate"] - first["BuildDate"]).dt.days

        gb = first.groupby(a.by, dropna=False)["delay_days"]
        if a.agg == "mean":
            res = gb.mean().reset_index(name="avg_delay_days")
            sort_col = "avg_delay_days"
        elif a.agg == "median":
            res = gb.median().reset_index(name="median_delay_days")
            sort_col = "median_delay_days"
        elif a.agg == "min":
            res = gb.min().reset_index(name="min_delay_days")
            sort_col = "min_delay_days"
        else:
            res = gb.max().reset_index(name="max_delay_days")
            sort_col = "max_delay_days"

        res = res.sort_values(sort_col, ascending=not a.sort_desc).head(a.limit)
        return _cap_df(res, max_rows_returned), None

    return pd.DataFrame([]), "Unsupported analysis plan kind."
