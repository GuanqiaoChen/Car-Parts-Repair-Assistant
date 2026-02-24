from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any

from .schema import (
    AssistantPlan,
    SimpleGroupbyPlan, PivotPlan, TrendPlan, FirstRepairDelayPlan,
    TopNSharePlan, DrilldownPlan, CorrelationPlan
)

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
    return df.head(max_rows) if max_rows and max_rows > 0 else df

def _maybe_row_count(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.ones(len(df), dtype=int), index=df.index)

def run_single_plan(df: pd.DataFrame, plan: AssistantPlan, max_rows_returned: int, max_series: int) -> Tuple[pd.DataFrame, Optional[str], Dict[str, Any]]:
    a = plan.analysis
    meta: Dict[str, Any] = {}

    if isinstance(a, SimpleGroupbyPlan):
        dff = _apply_filters(df, a.filters)

        if not a.groupby:
            out = {}
            if not a.metrics:
                out["row_count"] = int(len(dff))
                return pd.DataFrame([out]), None, meta

            for m in a.metrics:
                series = _maybe_row_count(dff) if m.col == "row_count" else dff[m.col]
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
            return pd.DataFrame([out]), None, meta

        gb = dff.groupby(list(a.groupby), dropna=False)

        if not a.metrics:
            res = gb.size().reset_index(name="row_count")
        else:
            agg_dict = {}
            for m in a.metrics:
                alias = m.alias or f"{m.agg}_{m.col}"
                if m.col == "row_count":
                    agg_dict[alias] = ("VIN", "count") if "VIN" in dff.columns else (dff.columns[0], "count")
                else:
                    agg_dict[alias] = (m.col, m.agg if m.agg != "median" else "median")
            res = gb.agg(**agg_dict).reset_index()

        if a.sort is not None and a.sort.by in res.columns:
            res = res.sort_values(by=a.sort.by, ascending=a.sort.ascending)

        res = res.head(max(1, int(a.limit)))
        return _cap_df(res, max_rows_returned), None, meta

    if isinstance(a, PivotPlan):
        dff = _apply_filters(df, a.filters).copy()

        if a.value == "row_count":
            dff["_row_count"] = 1
            value_col = "_row_count"
            aggfunc = "sum"
        else:
            value_col = "Quantity"
            aggfunc = "sum" if a.agg == "sum" else "count"

        piv = pd.pivot_table(
            dff,
            index=a.rows,
            columns=a.cols,
            values=value_col,
            aggfunc=aggfunc,
            fill_value=0,
        )

        piv = piv.iloc[: a.limit_rows, : a.limit_cols]
        return _cap_df(piv.reset_index(), max_rows_returned), None, meta

    if isinstance(a, TrendPlan):
        dff = _apply_filters(df, a.filters).copy()

        if a.date_col not in dff.columns:
            return pd.DataFrame([]), f"Missing date column: {a.date_col}", meta

        dff = dff.dropna(subset=[a.date_col])
        if dff.empty:
            return pd.DataFrame([]), "No rows with valid dates after filtering.", meta

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
            totals = dff.groupby(a.groupby)[val_col].agg(agg).sort_values(ascending=False).head(min(max_series, a.limit_series)).index
            dff = dff[dff[a.groupby].isin(totals)]
            res = dff.groupby(["_bucket", a.groupby], dropna=False)[val_col].agg(agg).reset_index(name="value")

        res = res.sort_values("_bucket").head(min(max_rows_returned, a.limit_points))
        return _cap_df(res, max_rows_returned), None, meta

    if isinstance(a, FirstRepairDelayPlan):
        dff = _apply_filters(df, a.filters).dropna(subset=["VIN", "DemandDate", "BuildDate"]).copy()
        if dff.empty:
            return pd.DataFrame([]), "No valid rows after filtering for delay computation.", meta

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
        return _cap_df(res, max_rows_returned), None, meta

    if isinstance(a, TopNSharePlan):
        dff = _apply_filters(df, a.filters).copy()
        if a.metric == "row_count":
            dff["_row_count"] = 1
            val_col = "_row_count"
            agg = "sum"
        else:
            val_col = "Quantity"
            agg = "sum" if a.agg == "sum" else "count"

        grouped = dff.groupby(a.dim, dropna=False)[val_col].agg(agg).reset_index(name="value")
        total = float(grouped["value"].sum()) if len(grouped) else 0.0

        grouped = grouped.sort_values("value", ascending=not a.sort_desc)
        top = grouped.head(a.top_n).copy()
        if a.include_other and len(grouped) > a.top_n:
            other_val = float(grouped.iloc[a.top_n:]["value"].sum())
            top = pd.concat([top, pd.DataFrame([{a.dim: "Other", "value": other_val}])], ignore_index=True)

        top["share"] = top["value"] / total if total > 0 else 0.0
        return _cap_df(top, max_rows_returned), None, meta

    if isinstance(a, DrilldownPlan):
        dff = _apply_filters(df, a.filters).copy()

        if a.top_metric == "row_count":
            dff["_row_count"] = 1
            val_col = "_row_count"
            top_agg = "sum"
        else:
            val_col = "Quantity"
            top_agg = "sum" if a.top_agg == "sum" else "count"

        top_df = dff.groupby(a.top_dim, dropna=False)[val_col].agg(top_agg).reset_index(name="value")
        top_df = top_df.sort_values("value", ascending=False).head(a.top_n)

        if top_df.empty:
            return pd.DataFrame([]), "No rows to compute drilldown after filtering.", meta

        top_value = top_df.iloc[0][a.top_dim]
        meta["top_dim"] = a.top_dim
        meta["top_value"] = top_value

        dff2 = dff[dff[a.top_dim] == top_value].copy()

        if a.breakdown_metric == "row_count":
            dff2["_row_count2"] = 1
            b_val = "_row_count2"
            b_agg = "sum"
        else:
            b_val = "Quantity"
            b_agg = "sum" if a.breakdown_agg == "sum" else "count"

        br = dff2.groupby(a.breakdown_dim, dropna=False)[b_val].agg(b_agg).reset_index(name="value")
        br = br.sort_values("value", ascending=False).head(a.breakdown_limit)

        br.insert(0, f"selected_{a.top_dim}", str(top_value))
        return _cap_df(br, max_rows_returned), None, meta

    if isinstance(a, CorrelationPlan):
        dff = _apply_filters(df, a.filters).copy()
        if len(dff) > a.sample:
            dff = dff.sample(n=a.sample, random_state=42)

        cols = [a.x, a.y]
        for c in cols:
            if c not in dff.columns:
                return pd.DataFrame([]), f"Missing column for correlation: {c}", meta

        tmp = dff[cols].dropna()
        if len(tmp) < 3:
            return pd.DataFrame([]), "Not enough non-null samples for correlation.", meta

        corr = float(tmp.corr(method="spearman" if a.method == "spearman" else "pearson").iloc[0, 1])
        meta["corr"] = corr
        meta["scatter_points"] = tmp.reset_index(drop=True).to_dict(orient="list")

        summary = pd.DataFrame([{"method": a.method, "x": a.x, "y": a.y, "correlation": corr, "n": int(len(tmp))}])
        return _cap_df(summary, max_rows_returned), None, meta

    return pd.DataFrame([]), "Unsupported analysis plan kind.", meta
