import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

load_dotenv()

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DATA_PATH = os.getenv("DATA_PATH", "/data/repair_history.csv")

app = FastAPI(title="Car Assistant Backend", version="0.1.0")

_df = None

class QueryRequest(BaseModel):
    question: str

@app.on_event("startup")
def startup():
    global _df
    _df = pd.read_csv(DATA_PATH)

@app.get("/health")
def health():
    return {"ok": True, "data_loaded": _df is not None}

@app.post("/query")
def query(req: QueryRequest):
    """
    Minimal stub endpoint:
    - echoes the question
    - returns a tiny demo table
    Replace this later with: LLM->plan->execute->table/chart.
    """
    demo_table = {
        "columns": ["field", "value"],
        "rows": [
            ["question", req.question],
            ["data_loaded", str(_df is not None)],
            ["num_rows", str(len(_df)) if _df is not None else "0"],
        ],
    }

    return {
        "answer_format": "table",
        "narrative": "Stub response (plumbing check).",
        "text": None,
        "table": demo_table,
        "chart": None,
        "plan": {
            "kind": "stub",
            "note": "Replace with LLM plan + Pandas execution",
        },
    }