from __future__ import annotations

import os
from openai import OpenAI
from .schema import AssistantPlan, DATA_COLUMNS

def system_prompt() -> str:
    cols = ", ".join(DATA_COLUMNS + ["VehicleAgeDays"])
    return f"""You are a data analyst planner.

You will receive a natural-language question about a CSV dataset with these columns:
{cols}

Your task: produce a valid JSON object matching the AssistantPlan schema.
Rules:
- Never invent columns outside the schema.
- Do NOT write code. Only output a plan.
- Prefer these analysis kinds:
  - simple_groupby: distributions, rankings, counts, sums, averages
  - pivot: 2D cross tab like "distribution across X and Y"
  - trend: time series over DemandDate (or BuildDate if asked)
  - first_repair_delay: time between BuildDate and FIRST DemandDate per VIN
- For "distribution": use row_count grouped by requested dimensions.
- For "most common" / "top": sort descending and limit 10.
- Choose answer_format:
  - chart for distributions/comparisons/trends
  - table for detailed breakdown
  - text only for short summaries
- chart.type suggestions:
  - bar for rankings
  - line for trend
  - heatmap for pivot
Return JSON only.
""".strip()

class LLMPlanner:
    def __init__(self) -> None:
        self.client = OpenAI()
        self.model = os.getenv("OPENAI_MODEL", "gpt-5.2")

    def plan(self, question: str) -> AssistantPlan:
        # Structured Outputs parsing into a Pydantic model
        resp = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": question},
            ],
            text_format=AssistantPlan,
        )
        return resp.output_parsed
