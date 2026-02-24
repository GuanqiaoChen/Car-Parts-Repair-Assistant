from __future__ import annotations

from typing import List
from .schema import AssistantPlan, MultiAssistantPlan
from .schema import (
    SimpleGroupbyPlan, PivotPlan, TrendPlan, FirstRepairDelayPlan,
    TopNSharePlan, DrilldownPlan, CorrelationPlan
)

def validate_assistant_plan(plan: AssistantPlan, max_rows_returned: int, max_series: int) -> List[str]:
    errs: List[str] = []
    a = plan.analysis

    if isinstance(a, SimpleGroupbyPlan):
        if a.limit > max_rows_returned:
            errs.append(f"simple_groupby.limit too large (>{max_rows_returned}).")
        if len(a.groupby) > 3:
            errs.append("simple_groupby.groupby supports up to 3 dimensions.")
        if len(a.metrics) > 3:
            errs.append("simple_groupby.metrics supports up to 3 metrics.")

    if isinstance(a, PivotPlan):
        if a.limit_rows > 50 or a.limit_cols > 50:
            errs.append("pivot size too large (limit_rows/cols should be <= 50).")

    if isinstance(a, TrendPlan):
        if a.limit_points > max_rows_returned:
            errs.append(f"trend.limit_points too large (>{max_rows_returned}).")
        if a.limit_series > max_series:
            errs.append(f"trend.limit_series too large (>{max_series}).")

    if isinstance(a, FirstRepairDelayPlan):
        if a.limit > max_rows_returned:
            errs.append(f"first_repair_delay.limit too large (>{max_rows_returned}).")

    if isinstance(a, TopNSharePlan):
        if a.top_n > 50:
            errs.append("topn_share.top_n too large (<= 50).")

    if isinstance(a, DrilldownPlan):
        if a.top_n > 10:
            errs.append("drilldown.top_n too large (<= 10).")
        if a.breakdown_limit > max_rows_returned:
            errs.append(f"drilldown.breakdown_limit too large (>{max_rows_returned}).")

    if isinstance(a, CorrelationPlan):
        if a.sample > 20000:
            errs.append("correlation.sample too large (<= 20000).")

    return errs

def validate_multi(plan: MultiAssistantPlan, max_rows_returned: int, max_series: int) -> List[str]:
    errs: List[str] = []
    if len(plan.plans) > 5:
        errs.append("Too many sub-plans (max 5).")
    for i, p in enumerate(plan.plans):
        sub_errs = validate_assistant_plan(p, max_rows_returned, max_series)
        errs.extend([f"plan[{i}]: {e}" for e in sub_errs])
    return errs
