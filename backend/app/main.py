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

app = FastAPI(title="Car Repair LLM Data Analyst Agent", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.getenv("DATA_PATH", "/data/encoded_parts_history.csv")
MAX_ROWS_RETURNED = int(os.getenv("MAX_ROWS_RETURNED", "200"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))
MAX_SERIES = int(os.getenv("MAX_SERIES", "10"))

df = None
planner = LLMPlanner()
plan_cache = TTLCache(ttl_seconds=CACHE_TTL_SECONDS)
result_cache = TTLCache(ttl_seconds=CACHE_TTL_SECONDS)

@app.on_event("startup")
def _startup():
    global df
    df = load_dataset(DATA_PATH)

@app.get("/health")
def health():
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

_ALLOWED_COLS = {
    "DemandDate","BuildDate","Province","CountryCode","Quantity","SSPart",
    "RequestType","DemandType","PayType","VehicleModel","VIN","VehicleAgeDays"
}

_ANALYTICS_HINTS = {
    "top","most","distribution","breakdown","trend","over time","monthly","weekly","daily",
    "count","sum","average","mean","median","min","max","correlation","scatter","heatmap","pivot",
    "by","compare","share","percentage","percent","list","unique"
}

def _is_gibberish_or_non_analytic(q: str) -> bool:
    s = q.strip().lower()
    if len(s) < 4:
        return True
    # if it has no analytics intent hints, treat as unclear
    return not any(h in s for h in _ANALYTICS_HINTS)

def _detect_unknown_field_mentions(q: str) -> str | None:
    """
    Return an error message if question appears to request an unsupported column.
    We detect:
    - CamelCase tokens like DealerName
    - explicit 'dealer' phrase
    """
    s = q.strip()
    s_lower = s.lower()

    synonyms = _load_synonyms_map()
    # allowed includes schema columns + synonyms keys
    allowed_tokens = set(_ALLOWED_COLS) | set(synonyms.keys()) | {"VIN"}  # VIN is allowed

    # CamelCase token detector: DealerName, CountryCode, VehicleModel...
    camel_tokens = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", s)
    for tok in camel_tokens:
        if tok not in allowed_tokens:
            return f"{tok} is not a column in this dataset. Please use one of: {', '.join(sorted(_ALLOWED_COLS))}."

    # common lowercase unsupported field mention
    if "dealer" in s_lower and "dealer" not in json.dumps(synonyms, ensure_ascii=False).lower():
        return f"DealerName (dealer) is not a column in this dataset. Please use one of: {', '.join(sorted(_ALLOWED_COLS))}."

    return None

def _apply_truncation_messaging(narr: str, explanation: str, meta: dict) -> tuple[str, str]:
    if not meta.get("truncated"):
        return narr, explanation
    note = meta.get("truncate_note") or "Full result is too large to return safely. Showing a truncated preview."
    # Put the strong statement in narrative, and details in explanation
    narr2 = f"⚠️ {note}"
    expl2 = f"{explanation}\n\nTruncation: {note}"
    return narr2, expl2

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    global df
    question = req.question.strip()

    # --- Fail 1: empty question ---
    if not question:
        return _friendly_fail("Please provide a non-empty question.")
    
    # --- Fail 2: non-analytic / gibberish ---
    if _is_gibberish_or_non_analytic(question):
        return _friendly_fail(
            "The request is unclear or not an analytical question. "
            "Please ask for a distribution/ranking/trend, e.g., 'top provinces by requests' or 'monthly trend of Quantity'."
        )

    # --- Fail 3: unknown column mention (e.g., DealerName) ---
    unknown_msg = _detect_unknown_field_mentions(question)
    if unknown_msg:
        return _friendly_fail(unknown_msg)

    plan_key = stable_hash({"q": question})
    multi = plan_cache.get(plan_key)
    if multi is None:
        try:
            multi = planner.plan_multi(question)
        except Exception as e:
            return _friendly_fail(f"Planner failed to produce a valid plan. Error: {e}")
        plan_cache.set(plan_key, multi)

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

        res_df, err, meta = run_single_plan(df, p, max_rows_returned=MAX_ROWS_RETURNED, max_series=MAX_SERIES)
        explanation = explain_plan(p)

        # --- 如果执行失败：保持原样（text item） ---
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

        # --- 成功执行：把“截断说明”写入 narrative / explanation ---
        narrative = p.narrative or ""
        narrative, explanation = _apply_truncation_messaging(narrative, explanation, meta)

        fmt = choose_output_format(p.answer_format, res_df)
        table = df_to_table(res_df)

        chart_json = None
        if fmt == "chart":
            chart_spec = p.chart if p.chart.type != "none" else choose_default_chart(res_df)
            chart_json = make_chart(res_df, chart_spec, meta=meta)
            if chart_json is None:
                fmt = "table"

        if p.analysis.kind == "correlation" and fmt != "chart":
            # try scatter by default
            chart_spec = p.chart
            if chart_spec.type == "none":
                chart_spec = type(p.chart)(type="scatter", title="Correlation scatter")
            chart_try = make_chart(res_df, chart_spec, meta=meta)
            if chart_try is not None:
                chart_json = chart_try
                fmt = "chart"

        item = ResultItem(
            answer_format=fmt,
            narrative=narrative,            # 用带截断提示的 narrative
            explanation=explanation,        # 用带截断提示的 explanation
            text=None,
            table=table if fmt in ["table", "chart"] else None,
            chart=chart_json if fmt == "chart" else None,
            plan=p_dump,
        )
        items.append(item)
        result_cache.set(res_key, item.model_dump())

    final_narr = multi.final_narrative or "Completed the requested analyses."

    # --- 如果任何 step 发生截断：返回建议，否则 suggestions 为空 ---
    any_truncated = any(
        (getattr(it, "narrative", "") or "").startswith("⚠️")
        for it in items
    )
    sugg = default_suggestions() if any_truncated else []

    return QueryResponse(items=items, final_narrative=final_narr, suggestions=sugg)
