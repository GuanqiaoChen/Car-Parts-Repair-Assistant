from __future__ import annotations

import pandas as pd
import plotly.express as px
from typing import Optional, Dict, Any

from .schema import ChartSpec

def make_chart(df: pd.DataFrame, chart: ChartSpec) -> Optional[Dict[str, Any]]:
    if chart.type == "none" or df is None or df.empty:
        return None

    cols = list(df.columns)
    x = chart.x or (cols[0] if len(cols) > 0 else None)
    y = chart.y or (cols[1] if len(cols) > 1 else None)
    color = chart.color
    title = chart.title or ""

    if x is None:
        return None

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
        # If pivot plan returns a wide table (rows dim + many cols), we can melt it for a heatmap.
        if len(cols) < 3:
            return None
        row_dim = cols[0]
        melted = df.melt(id_vars=[row_dim], var_name="col_dim", value_name="value")
        fig = px.density_heatmap(melted, x="col_dim", y=row_dim, z="value", title=title)
    else:
        return None

    return fig.to_dict()
