# Car Parts Repair LLM Data Analyst Assistant

A project that builds an LLM-powered data analyst assistant for an encoded car parts repair history dataset. The system accepts natural-language queries, converts them into a safe structured analysis plan, executes the plan deterministically in Pandas, and returns results as text, tables, and charts.

This repo includes:
- Backend: FastAPI service for planning + validation + execution
- Frontend: Streamlit UI for interactive querying and result display
- Docker: docker-compose for startup
- Safety/Robustness: Schema-based planning, guardrails, truncation previews, and dynamic suggestions on failures


## Architecture Overview

User Question -> LLM Planner -> Structured Plan -> Safe Pandas Execution -> Rendered Results

1) LLM Planner (Structured Output)
Converts the user's natural-language query into a JSON plan that matches strict Pydantic schemas. This reduces hallucinations and prevents arbitrary code generation.

2) Guardrails / Validation
Plans are validated to keep operations bounded and predictable (e.g., pivot size limits, series caps, sampling caps).

3) Deterministic Execution
The backend executes only approved operations (groupby, pivot, trend bucketing, first-repair-delay, topN+share, drilldown, correlation), all implemented in Pandas.

4) UI Signals (Product-like)
Each step returns:
- status="ok" for normal results
- status="preview" when the backend must return a truncated preview (system-level truncation)
- status="error" when the request cannot be executed as requested
The frontend renders preview as warnings and error as errors.


## Features Supported

- Distributions / rankings (counts, sums, means)
- Top-N + share-of-total
- Drilldown: find top group, then breakdown within that group
- Pivot / heatmap (2D cross-tabs with safety caps)
- Trends over time (daily/weekly/monthly) with optional series split
- First repair delay: BuildDate → first DemandDate per VIN, aggregated by dimension
- Correlation (pearson/spearman) with optional sampling for performance
- Synonyms mapping (e.g., “state” → Province, “model” → VehicleModel)
- Dynamic suggestions only when needed (execution error or system-level truncation)


## Project Structure

```text
.
├─ backend/
│  ├─ app/
│  │  ├─ main.py          # FastAPI routes /query, /health
│  │  ├─ schema.py        # Pydantic schemas for plans + response
│  │  ├─ llm.py           # LLM planner (structured outputs)
│  │  ├─ execute.py       # Deterministic Pandas execution engine
│  │  ├─ viz.py           # Plotly chart generation (JSON-safe)
│  │  ├─ suggest.py       # Dynamic suggestion generation
│  │  ├─ data.py          # Dataset loading + preprocessing
│  │  └─ resources/
│  │     └─ synonyms.json # Synonyms mapping used by planner
│  ├─ requirements.txt
│  ├─ Dockerfile
│  └─ .env.example
├─ frontend/
│  ├─ app.py              # Streamlit UI
│  ├─ requirements.txt
│  ├─ Dockerfile
│  └─ .env.example
├─ data/                  # Put CSV/Data files here (not committed)
├─ docker-compose.yml
└─ README.md
```

## Prerequisites

- Docker Desktop (recommended)
- OR Python 3.11 locally (if running without Docker)
- An OpenAI API key


## Setup (Docker)

1) Place dataset
Put the data file in:
data/parts_history.csv

2) Create env files
Copy examples:
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env

Edit backend/.env and set:
- OPENAI_API_KEY=...
- DATA_PATH=/data/encoded_parts_history.csv
- (optional) OPENAI_MODEL=...

3) Start services
From repo root:
docker compose up -d --build

Open:
- Streamlit UI: http://localhost:8501
- FastAPI docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health


## Setup (Local)

Backend:
- cd backend
- python -m venv .venv
- (activate venv)
- pip install -r requirements.txt

- cp .env.example .env
- (edit .env, including OPENAI_API_KEY and DATA_PATH)

- uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Frontend:
- cd frontend
- python -m venv .venv
- (activate venv)
- pip install -r requirements.txt

- cp .env.example .env
- (edit .env: API_BASE_URL=http://localhost:8000)

- streamlit run app.py


## How to Use

1) Open Streamlit UI at http://localhost:8501
2) Enter a natural-language question
3) The UI displays results as one or more steps, each with:
- a narrative (intent)
- an explanation (what was computed)
- optional text/table/chart output
- optional notices (warnings/errors)

Status meanings
- ok: step executed normally
- preview: backend returned a truncated preview due to system-level safety/performance limits
- error: step could not be executed as requested


## Example Queries

Basic counts / rankings
- How many repair requests are in the dataset?
- Top 10 Provinces by number of repair requests.
- Top 10 VehicleModel by total Quantity requested.

Distributions
- Show the distribution of DemandType by row count.
- Show PayType distribution by row count.

Drilldown (dependency handled in one plan)
- Which province has the most requests, and within that province what is the most common PayType?
- Find the VehicleModel with the most requests, then show its DemandType distribution.

Pivot / heatmap
- Make a heatmap of DemandType by VehicleModel using row counts.
- Cross-tab PayType by Province using row counts (heatmap).

Trends
- Show monthly trend of request count.
- Weekly trend of total Quantity requested.
- Monthly request trend split by CountryCode (top series only).

First repair delay
- Average time between BuildDate and first DemandDate by VehicleModel.
- Median time to first repair by Province.

Correlation
- Is vehicle age correlated with Quantity? Use a scatter plot.
- Compute Spearman correlation between Quantity and VehicleAgeDays.


## Notes on Safety Limits and Previews

Some requests can produce extremely large outputs (e.g., listing all VINs, or a VIN×SSPart heatmap). In those cases, the backend will return a preview with:
- status="preview"
- a warning notice describing the truncation reason

To get more specific outputs, narrow the scope:
- add filters (CountryCode / VehicleModel / Province)
- ask for top N results
- ask for 1D distributions instead of full pivots


## Troubleshooting

Backend cannot find dataset
If you see file-not-found errors, verify:
- the CSV exists at data/encoded_parts_history.csv
- backend/.env contains DATA_PATH=/data/encoded_parts_history.csv

Refresh after changes
If code changes do not reflect, restart backend:
docker compose restart backend

Or force recreate (if compose config/env changes):
docker compose up -d --force-recreate

Check service health
- http://localhost:8000/health
- In Streamlit sidebar, click “Health check”


## Notes

- data/ and .env files are not committed.
- The repo includes .env.example templates and full instructions to run locally or via Docker.
- The system is designed to handle a broad range of user-defined questions dynamically, while enforcing safe execution boundaries.

## Contribution
Guanqiao Chen
