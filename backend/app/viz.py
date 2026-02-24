from __future__ import annotations

import json
import pandas as pd
import plotly.express as px
from typing import Optional, Dict, Any

from .schema import ChartSpec

def choose_default_chart(df: pd.DataFrame) -> ChartSpec:
    cols = list(df.columns)
    if len(cols) >= 2:
        return ChartSpec(type="bar", x=cols[0], y=cols[1])
    return ChartSpec(type="none")

def make_chart(df: pd.DataFrame, chart: ChartSpec, meta: Optional[dict] = None) -> Optional[Dict[str, Any]]:
    if df is None or df.empty:
        return None

    if chart.type == "scatter" and meta and "scatter_points" in meta:
        pts = meta["scatter_points"]
        keys = list(pts.keys())
        if len(keys) >= 2:
            xcol, ycol = keys[0], keys[1]
            tmp = pd.DataFrame({xcol: pts[xcol], ycol: pts[ycol]})
            fig = px.scatter(tmp, x=xcol, y=ycol, title=chart.title or "Correlation scatter")
            return json.loads(fig.to_json())

    if chart.type == "none":
        return None

    cols = list(df.columns)
    x = chart.x or (cols[0] if len(cols) > 0 else None)
    y = chart.y or (cols[1] if len(cols) > 1 else None)
    color = chart.color
    title = chart.title or ""

    if x is None:
        return None

    try:
        if chart.type == "bar":
            if y is None or y not in df.columns:
                return None
            fig = px.bar(df, x=x, y=y, color=color, title=title)
        elif chart.type == "line":
            if y is None or y not in df.columns:
                return None
            fig = px.line(df, x=x, y=y, color=color, title=title)
        elif chart.type == "pie":
            if y is None or y not in df.columns:
                return None
            fig = px.pie(df, names=x, values=y, title=title)
        elif chart.type == "heatmap":
            if len(cols) < 3:
                return None
            row_dim = cols[0]
            melted = df.melt(id_vars=[row_dim], var_name="col_dim", value_name="value")
            fig = px.density_heatmap(melted, x="col_dim", y=row_dim, z="value", title=title)
        else:
            return None
        return json.loads(fig.to_json())
    except Exception:
        return None
