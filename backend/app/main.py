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
from .suggestions import default_suggestions
from .cache import TTLCache, stable_hash

import re
import json
from pathlib import Path


load_dotenv()

# FastAPI surface for the “ask in natural language → get analysis” workflow.
# All heavy lifting (planning, execution, formatting) happens in collaborators;
# this module is responsible for wiring them together and enforcing guardrails.
app = FastAPI(title="Car Repair LLM Data Analyst Agent", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global knobs that shape how much data the agent is allowed to touch and return.
DATA_PATH = os.getenv("DATA_PATH", "/data/encoded_parts_history.csv")
MAX_ROWS_RETURNED = int(os.getenv("MAX_ROWS_RETURNED", "200"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))
MAX_SERIES = int(os.getenv("MAX_SERIES", "10"))

# Process‑wide state: loaded dataset plus an LLM planner and small in‑memory caches
# so repeated questions and plans are fast and deterministic.
df = None
planner = LLMPlanner()
plan_cache = TTLCache(ttl_seconds=CACHE_TTL_SECONDS)
result_cache = TTLCache(ttl_seconds=CACHE_TTL_SECONDS)

@app.on_event("startup")
def _startup():
    """Load and pre‑process the dataset once when the service boots."""
    global df
    df = load_dataset(DATA_PATH)

@app.get("/health")
def health():
    """Lightweight probe used by the UI and infra to check readiness."""
    return {"ok": True, "data_loaded": df is not None, "rows": (len(df) if df is not None else 0)}


def _friendly_fail(message: str) -> QueryResponse:
    return QueryResponse(
        items=[ResultItem(
            answer_format="text",
            narrative="I couldn't run that request as-is.",
            explanation="The request was out of the supported safe operations or was too ambiguous.",
            text=message,
            table=None,
            chart=None,
            plan={"kind": "error"},
        )],
        final_narrative="Request not executed.",
        suggestions=default_suggestions(),
    )


def _load_synonyms_map() -> dict:
    p = Path(__file__).parent / "resources" / "synonyms.json"
    return json.loads(p.read_text(encoding="utf-8"))

# Whitelist of data fields the planner and executor are allowed to touch.
_ALLOWED_COLS = {
    "DemandDate","BuildDate","Province","CountryCode","Quantity","SSPart",
    "RequestType","DemandType","PayType","VehicleModel","VIN","VehicleAgeDays"
}

# A loose set of words that indicate the user is asking for analysis instead of
# free‑form chit‑chat; used as an early guardrail before we ever hit the LLM.
_ANALYTICS_HINTS = {
    "top","most","distribution","breakdown","trend","over time","monthly","weekly","daily",
    "count","sum","average","mean","median","min","max","correlation","scatter","heatmap","pivot",
    "by","compare","share","percentage","percent","list","unique"
}


def _is_gibberish_or_non_analytic(q: str) -> bool:
    s = q.strip().lower()
    if len(s) < 4:
        return True
    # If it has no analytics intent hints, treat as unclear and keep the model idle.
    return not any(h in s for h in _ANALYTICS_HINTS)

def _detect_unknown_field_mentions(q: str) -> str | None:
    """
    Defensive pass over the raw question before we touch the LLM.

    The goal is to catch obvious references to columns that do not exist
    (for example a `DealerName` field) and fail fast with a human‑readable
    message rather than letting the model hallucinate SQL or Pandas code.
    """
    s = q.strip()
    s_lower = s.lower()

    synonyms = _load_synonyms_map()
    # Allowed includes schema columns + synonyms keys so we can support
    # user‑friendly aliases without widening the actual execution surface.
    allowed_tokens = set(_ALLOWED_COLS) | set(synonyms.keys()) | {"VIN"}  # VIN is allowed

    # CamelCase token detector: DealerName, CountryCode, VehicleModel...
    camel_tokens = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", s)
    for tok in camel_tokens:
        if tok not in allowed_tokens:
            return f"{tok} is not a column in this dataset. Please use one of: {', '.join(sorted(_ALLOWED_COLS))}."

    # Common lowercase unsupported field mention.
    if "dealer" in s_lower and "dealer" not in json.dumps(synonyms, ensure_ascii=False).lower():
        return f"DealerName (dealer) is not a column in this dataset. Please use one of: {', '.join(sorted(_ALLOWED_COLS))}."

    return None

def _apply_truncation_messaging(narr: str, explanation: str, meta: dict) -> tuple[str, str]:
    if not meta.get("truncated"):
        return narr, explanation
    note = meta.get("truncate_note") or "Full result is too large to return safely. Showing a truncated preview."
    # Put the strong statement in the short narrative and leave details in the explanation;
    # this keeps the UI readable while still being honest about what was returned.
    narr2 = f"⚠️ {note}"
    expl2 = f"{explanation}\n\nTruncation: {note}"
    return narr2, expl2

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Main natural‑language entrypoint.

    This function is the top‑level orchestrator that:
    1) sanitises and lightly interprets the raw question,
    2) asks the LLM planner to turn it into an executable plan,
    3) validates and executes that plan on the Pandas DataFrame, and
    4) shapes the result into text, tables or charts for the frontend.
    """
    global df
    question = req.question.strip()

    # Fail 1: empty question
    if not question:
        return _friendly_fail("Please provide a non-empty question.")

    # Fail 2: non-analytic / gibberish
    # We do a cheap lexical scan before calling the LLM so obviously non‑analytic
    # prompts don't burn tokens or accidentally trigger complex plans.
    if _is_gibberish_or_non_analytic(question):
        return _friendly_fail(
            "The request is unclear or not an analytical question. "
            "Please ask for a distribution/ranking/trend, e.g., 'top provinces by requests' or 'monthly trend of Quantity'."
        )

    # Fail 3: unknown column mention (e.g., DealerName)
    # Here we still look at the raw text, but we lean on the synonyms map so that
    # user‑friendly phrases can be mapped onto the strict schema.
    unknown_msg = _detect_unknown_field_mentions(question)
    if unknown_msg:
        return _friendly_fail(unknown_msg)

    # Cache by normalized question so repeated queries reuse the same LLM‑generated
    # plan instead of hitting the model every time.
    plan_key = stable_hash({"q": question})
    multi = plan_cache.get(plan_key)
    if multi is None:
        try:
            multi = planner.plan_multi(question)
        except Exception as e:
            return _friendly_fail(f"Planner failed to produce a valid plan. Error: {e}")
        plan_cache.set(plan_key, multi)

    # Safety net around the LLM output: ensure the structured plan stays within
    # the bounds we are willing to execute (row counts, number of series, etc.).
    errs = validate_multi(multi, max_rows_returned=MAX_ROWS_RETURNED, max_series=MAX_SERIES)
    if errs:
        msg = "Plan validation failed:\n- " + "\n- ".join(errs)
        return _friendly_fail(msg)

    items = []
    for p in multi.plans:
        p_dump = p.model_dump()
        res_key = stable_hash(p_dump)
        cached = result_cache.get(res_key)
        if cached is not None:
            items.append(ResultItem(**cached))
            continue

        # Run the concrete plan on the pre‑processed DataFrame and capture any
        # truncation metadata the executor produces for downstream messaging.
        res_df, err, meta = run_single_plan(df, p, max_rows_returned=MAX_ROWS_RETURNED, max_series=MAX_SERIES)
        explanation = explain_plan(p)

        # If execution failed, surface the error as plain text so the user
        # can correct the question or we can adjust the planning heuristics.
        if err:
            item = ResultItem(
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

        # Successful execution: weave truncation notes into the human‑facing copy.
        narrative = p.narrative or ""
        narrative, explanation = _apply_truncation_messaging(narrative, explanation, meta)

        # Decide whether this step should come back as text, a table, or a chart.
        fmt = choose_output_format(p.answer_format, res_df)
        table = df_to_table(res_df)

        chart_json = None
        if fmt == "chart":
            # Prefer the chart shape specified by the planner, otherwise fall back
            # to a simple default that still makes the aggregation legible.
            chart_spec = p.chart if p.chart.type != "none" else choose_default_chart(res_df)
            chart_json = make_chart(res_df, chart_spec, meta=meta)
            if chart_json is None:
                fmt = "table"

        if p.analysis.kind == "correlation" and fmt != "chart":
            # Correlation plans are visually easier to reason about, so we try to
            # upgrade them to a scatter chart if possible.
            chart_spec = p.chart
            if chart_spec.type == "none":
                chart_spec = type(p.chart)(type="scatter", title="Correlation scatter")
            chart_try = make_chart(res_df, chart_spec, meta=meta)
            if chart_try is not None:
                chart_json = chart_try
                fmt = "chart"

        item = ResultItem(
            answer_format=fmt,
            narrative=narrative,
            explanation=explanation,
            text=None,
            table=table if fmt in ["table", "chart"] else None,
            chart=chart_json if fmt == "chart" else None,
            plan=p_dump,
        )
        items.append(item)
        result_cache.set(res_key, item.model_dump())

    final_narr = multi.final_narrative or "Completed the requested analyses."

    # If any step was truncated, we surface additional suggestion text so the user
    # knows how to refine the question to see more detail.
    any_truncated = any(
        (getattr(it, "narrative", "") or "").startswith("⚠️")
        for it in items
    )
    sugg = default_suggestions() if any_truncated else []

    return QueryResponse(items=items, final_narrative=final_narr, suggestions=sugg)
