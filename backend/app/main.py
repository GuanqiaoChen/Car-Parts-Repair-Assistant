from __future__ import annotations

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schema import QueryRequest, QueryResponse, ResultItem
from .data import load_dataset
from .llm import LLMPlanner
from .execute import run_single_plan
from .viz import make_chart, choose_default_chart
from .formatter import df_to_table, choose_output_format
from .validator import validate_multi
from .explain import explain_plan
from .cache import TTLCache, stable_hash

# Dynamic suggestions are computed from runtime context
from .suggest import build_suggestions

load_dotenv()

# HTTP service that hosts the full assistant pipeline:
# - planning (LLM -> structured plan),
# - validation (guardrails), and
# - execution (safe Pandas + chart rendering).
app = FastAPI(title="Car Repair LLM Data Analyst Agent", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Process-level settings. These set the default safety/latency envelope.
DATA_PATH = os.getenv("DATA_PATH", "/data/encoded_parts_history.csv")
MAX_ROWS_RETURNED = int(os.getenv("MAX_ROWS_RETURNED", "200"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))
MAX_SERIES = int(os.getenv("MAX_SERIES", "10"))

# Global state:
# - the dataset is loaded once on startup,
# - the planner is reused across requests, and
# - small TTL caches keep demos fast and stable.
df = None
planner = LLMPlanner()
plan_cache = TTLCache(ttl_seconds=CACHE_TTL_SECONDS)
result_cache = TTLCache(ttl_seconds=CACHE_TTL_SECONDS)


@app.on_event("startup")
def _startup():
    """
    Load the dataset once per process.

    The rest of the system assumes the DataFrame already has:
    - parsed dates, and
    - derived fields such as VehicleAgeDays.
    """
    global df
    df = load_dataset(DATA_PATH)


@app.get("/health")
def health():
    """
    Health probe for the frontend.

    It answers:
    - whether the API is reachable, and
    - whether the dataset has been loaded into memory.
    """
    return {"ok": True, "data_loaded": df is not None, "rows": (len(df) if df is not None else 0)}


def _dedupe(items: list[str]) -> list[str]:
    """
    Preserve order while removing duplicates.

    Suggestions are deliberately small and should not repeat.
    """
    out: list[str] = []
    seen = set()
    for x in items:
        k = x.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return out


def _friendly_fail(question: str, message: str, plan_kind: str = "error") -> QueryResponse:
    """
    Build a consistent "request not executed" response.

    This is used for:
    - planner failures,
    - validation failures, and
    - early request issues (e.g., empty question).
    """
    sugg = build_suggestions(
        question=question,
        plan_dump={"analysis": {"kind": plan_kind}},
        err=message,
        meta={},
    )
    return QueryResponse(
        items=[
            ResultItem(
                status="error",
                notices=[message],
                answer_format="text",
                narrative="I couldn't run that request as-is.",
                explanation="The request could not be planned or executed with the supported safe operations.",
                text=message,
                table=None,
                chart=None,
                plan={"kind": "error"},
            )
        ],
        final_narrative="Request not executed.",
        suggestions=_dedupe(sugg),
    )


def _system_notices(meta: dict) -> list[str]:
    """
    Translate system-level truncation metadata into UI notices.

    These notices are shown as warning banners in the frontend when the backend
    returns a preview rather than the full output.
    """
    if not meta.get("truncated"):
        return []
    note = meta.get("truncate_note") or "Full result is too large to return safely. Showing a truncated preview."
    return [note]


def _user_limit_note(meta: dict) -> str:
    """
    Plan/user-level limits are normal and should not be treated as warnings.

    Examples:
    - top 10,
    - limit_points, and
    - drilldown breakdown_limit.
    """
    if not meta.get("user_limited"):
        return ""
    return meta.get("user_limit_note") or "Result is limited by the requested top/limit setting."


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    End-to-end query endpoint.

    Steps:
    1. Plan: LLM converts the natural-language question to MultiAssistantPlan.
    2. Validate: guardrails prevent oversized or unsafe work.
    3. Execute: run each step deterministically (no arbitrary code execution).
    4. Render: return text/table/chart payloads that the frontend can display.
    """
    global df
    question = req.question.strip()
    if not question:
        return _friendly_fail(question, "Please provide a non-empty question.")

    # Plan cache: identical questions reuse the same plan within TTL.
    plan_key = stable_hash({"q": question})
    multi = plan_cache.get(plan_key)
    if multi is None:
        try:
            multi = planner.plan_multi(question)
        except Exception as e:
            return _friendly_fail(
                question,
                f"Planner failed to produce a valid plan. Error: {e}",
                plan_kind="planner_error",
            )
        plan_cache.set(plan_key, multi)

    # Guardrails: reject plans that exceed safe defaults.
    errs = validate_multi(multi, max_rows_returned=MAX_ROWS_RETURNED, max_series=MAX_SERIES)
    if errs:
        msg = "Plan validation failed:\n- " + "\n- ".join(errs)
        return _friendly_fail(question, msg, plan_kind="validation_error")

    items: list[ResultItem] = []
    dyn_suggestions: list[str] = []

    # Execute each step. Suggestions only appear when:
    # - a step fails, or
    # - a step returns a system-level truncated preview.
    for p in multi.plans:
        p_dump = p.model_dump()
        res_key = stable_hash(p_dump)

        cached = result_cache.get(res_key)
        if cached is not None:
            items.append(ResultItem(**cached))
            continue

        res_df, err, meta = run_single_plan(df, p, max_rows_returned=MAX_ROWS_RETURNED, max_series=MAX_SERIES)
        explanation = explain_plan(p)

        if err:
            # Failure: return an error step and attach dynamic suggestions.
            dyn_suggestions.extend(build_suggestions(
                question=question,
                plan_dump=p_dump,
                err=err,
                meta=meta,
            ))
            item = ResultItem(
                status="error",
                notices=[err],
                answer_format="text",
                narrative=p.narrative or "Unable to run the requested analysis.",
                explanation=explanation,
                text=err,
                table=None,
                chart=None,
                plan=p_dump,
            )
            items.append(item)
            result_cache.set(res_key, item.model_dump())
            continue

        if meta.get("truncated"):
            # Preview: the backend applied a system-level cap and is returning
            # a limited view. We attach dynamic suggestions for narrowing.
            dyn_suggestions.extend(build_suggestions(
                question=question,
                plan_dump=p_dump,
                err=None,
                meta=meta,
            ))

        # Non-warning note for plan/user-level limits.
        user_note = _user_limit_note(meta)
        if user_note:
            explanation = f"{explanation}\n\nNote: {user_note}"

        # Determine output format. Even if the plan requests a chart, the backend
        # may downgrade to a table if plotting is not feasible.
        fmt = choose_output_format(p.answer_format, res_df)
        table = df_to_table(res_df)

        chart_json = None
        if fmt == "chart":
            chart_spec = p.chart if p.chart.type != "none" else choose_default_chart(res_df)
            chart_json = make_chart(res_df, chart_spec, meta=meta)
            if chart_json is None:
                fmt = "table"

        # Correlation: if the plan did not request a chart, attempt a scatter by default.
        if p.analysis.kind == "correlation" and fmt != "chart":
            chart_spec = p.chart
            if chart_spec.type == "none":
                chart_spec = type(p.chart)(type="scatter", title="Correlation scatter")
            chart_try = make_chart(res_df, chart_spec, meta=meta)
            if chart_try is not None:
                chart_json = chart_try
                fmt = "chart"

        status = "preview" if meta.get("truncated") else "ok"
        notices = _system_notices(meta)

        item = ResultItem(
            status=status,
            notices=notices,
            answer_format=fmt,
            narrative=p.narrative or "",
            explanation=explanation,
            text=None,
            table=table if fmt in ["table", "chart"] else None,
            chart=chart_json if fmt == "chart" else None,
            plan=p_dump,
        )
        items.append(item)
        result_cache.set(res_key, item.model_dump())

    # The final narrative is a short wrap-up from the planner. When missing,
    # fall back to a conservative default.
    final_narr = multi.final_narrative or "Completed the requested analyses."
    return QueryResponse(
        items=items,
        final_narrative=final_narr,
        suggestions=_dedupe(dyn_suggestions),
    )