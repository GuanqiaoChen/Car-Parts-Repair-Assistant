from __future__ import annotations

from typing import List, Optional, Dict, Any

ALLOWED_DIMS = ["Province", "CountryCode", "VehicleModel", "DemandType", "PayType", "RequestType", "SSPart", "VIN"]
ALLOWED_DATE = ["DemandDate", "BuildDate"]
ALLOWED_METRICS = ["row_count", "Quantity", "VehicleAgeDays"]

def build_suggestions(
    *,
    question: str,
    plan_dump: Dict[str, Any],
    err: Optional[str],
    meta: Dict[str, Any],
) -> List[str]:
    """
    Dynamic suggestions. Suggestions are derived from runtime context.
    """
    s: List[str] = []
    kind = (plan_dump.get("analysis") or {}).get("kind")

    # Suggestions derived from execution/planning errors.
    if err:
        e = err.lower()

        if "missing date column" in e or "valid dates" in e:
            s.append(f"Try asking a trend explicitly over {ALLOWED_DATE[0]} (e.g., monthly trend).")
            s.append("If you need a date filter, specify a range like: between 2024-01-01 and 2024-06-30.")

        elif "missing column" in e:
            s.append("Use only dataset columns; try grouping by Province, VehicleModel, DemandType, PayType, or CountryCode.")
            s.append("If you meant a different concept, rephrase using one of the existing columns (e.g., 'state' -> Province).")

        elif "not enough" in e and "correlation" in e:
            s.append("Try removing strict filters to increase sample size, or correlate Quantity with VehicleAgeDays without additional filters.")
            s.append("If you want a distribution instead, ask for 'distribution of Quantity by VehicleModel'.")

        elif "unsupported" in e:
            s.append("Try asking for: (1) a distribution, (2) a top-N ranking, (3) a trend, or (4) a pivot heatmap.")
            s.append("Example: 'Top 10 Provinces by request count' or 'Heatmap of DemandType by VehicleModel'.")

        else:
            # Generic fallback that stays grounded in allowed dimensions.
            s.append(f"Try specifying a dimension to group by, such as {', '.join(ALLOWED_DIMS[:5])}.")
            s.append(f"Try specifying a metric, such as {', '.join(ALLOWED_METRICS)}.")

        return _dedupe(s)

    # Suggestions derived from system truncation context.
    if meta.get("truncated"):
        note = (meta.get("truncate_note") or "").lower()

        if kind == "pivot" or "pivot" in note or "heatmap" in note:
            s.append("Add filters (CountryCode, VehicleModel, Province) to reduce matrix size.")
            s.append("Ask for a smaller heatmap like 'top 10 VehicleModel by row count' first, then pivot within that subset.")
            s.append("Ask for a 1D distribution instead of a full pivot, e.g., 'DemandType distribution by row count'.")

        elif kind == "trend" or "series" in note:
            s.append("Limit the number of series: ask 'top 5 CountryCode series only' or remove the split dimension.")
            s.append("Filter to a single CountryCode or VehicleModel, then rerun the trend.")

        elif kind == "correlation" or "sample" in note:
            s.append("If you need exact correlation on all rows, ask for a filtered subset (e.g., one VehicleModel/CountryCode).")
            s.append("Alternatively, ask for summary stats (mean/median) by a dimension instead of correlation.")

        else:
            s.append("Add filters to narrow scope (CountryCode, Province, VehicleModel) or ask for top N results.")
            s.append("If you need the full detailed output, request an export workflow (not supported in this UI).")

        return _dedupe(s)

    # No error and no system truncation: return no suggestions.
    return []


def _dedupe(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        k = x.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return out
