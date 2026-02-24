from __future__ import annotations

from .schema import AssistantPlan
from .schema import (
    SimpleGroupbyPlan, PivotPlan, TrendPlan, FirstRepairDelayPlan,
    TopNSharePlan, DrilldownPlan, CorrelationPlan
)


def explain_plan(plan: AssistantPlan) -> str:
    a = plan.analysis
    if isinstance(a, SimpleGroupbyPlan):
        if a.groupby:
            metric_str = ", ".join([f"{m.agg}({m.col}) as {m.alias}" for m in a.metrics]) if a.metrics else "row_count"
            return f"Grouped by {list(a.groupby)}, computed {metric_str}, then applied sort/limit."
        return "Computed an overall aggregation without grouping."
    if isinstance(a, PivotPlan):
        return f"Created a 2D pivot with rows={a.rows}, cols={a.cols}, value={a.value}, agg={a.agg}."
    if isinstance(a, TrendPlan):
        gb = f" split by {a.groupby}" if a.groupby else ""
        return f"Bucketed {a.date_col} by freq={a.freq}{gb}, aggregated {a.metric} with {a.agg}."
    if isinstance(a, FirstRepairDelayPlan):
        return f"Computed first repair delay per VIN (first DemandDate - BuildDate), then aggregated delay_days by {a.by} using {a.agg}."
    if isinstance(a, TopNSharePlan):
        return f"Computed top {a.top_n} for {a.dim} using {a.agg}({a.metric}), and calculated share of total."
    if isinstance(a, DrilldownPlan):
        return f"Found top {a.top_n} {a.top_dim} by {a.top_agg}({a.top_metric}), then within the top value computed {a.breakdown_agg}({a.breakdown_metric}) by {a.breakdown_dim}."
    if isinstance(a, CorrelationPlan):
        return f"Computed {a.method} correlation between {a.x} and {a.y} on a sampled subset (n≤{a.sample})."
    # Fallback for forwards‑compatibility: if we add a new analysis kind and
    # forget to update this helper, the UI still shows a sane description.
    return "Executed the planned analysis."
