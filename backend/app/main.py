from __future__ import annotations

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schema import QueryRequest, QueryResponse
from .data import load_dataset
from .llm import LLMPlanner
from .execute import run_plan
from .viz import make_chart

load_dotenv()

app = FastAPI(title="Car Repair LLM Data Analyst Agent", version="1.0.0")

# For local dev + Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.getenv("DATA_PATH", "/data/encoded_parts_history.csv")
MAX_ROWS_RETURNED = int(os.getenv("MAX_ROWS_RETURNED", "200"))

df = None
planner = LLMPlanner()

@app.on_event("startup")
def _startup():
    global df
    df = load_dataset(DATA_PATH)

@app.get("/health")
def health():
    return {"ok": True, "data_loaded": df is not None, "rows": (len(df) if df is not None else 0)}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    global df
    plan = planner.plan(req.question)

    result_df, error_text = run_plan(df, plan, max_rows_returned=MAX_ROWS_RETURNED)

    if error_text:
        return QueryResponse(
            answer_format="text",
            narrative=plan.narrative or "Unable to run the requested analysis.",
            text=error_text,
            table=None,
            chart=None,
            plan=plan.model_dump(),
        )

    table_json = {
        "columns": [str(c) for c in result_df.columns],
        "rows": result_df.astype(object).where(result_df.notna(), None).values.tolist(),
    }

    chart_json = None
    if plan.answer_format == "chart":
        chart_json = make_chart(result_df, plan.chart)

    # If chart requested but we failed to form one, still return table
    answer_format = plan.answer_format
    if answer_format == "chart" and chart_json is None:
        answer_format = "table"

    return QueryResponse(
        answer_format=answer_format,
        narrative=plan.narrative or "",
        text=None,
        table=table_json if answer_format in ["table", "chart"] else None,
        chart=chart_json if answer_format == "chart" else None,
        plan=plan.model_dump(),
    )
