from __future__ import annotations

import json
from pathlib import Path
import os
from openai import OpenAI

from .schema import MultiAssistantPlan, DATA_COLUMNS

def _load_synonyms() -> dict:
    """
    Load the hand‑curated mapping between user‑facing phrases and strict
    dataset column names. This is the main bridge between natural language
    questions and the schema the executor is allowed to touch.
    """
    p = Path(__file__).parent / "resources" / "synonyms.json"
    return json.loads(p.read_text(encoding="utf-8"))

def system_prompt() -> str:
    """
    Build the single source of truth for how we talk to the LLM planner.

    The prompt carefully describes:
    - which columns exist and how they can be referenced,
    - how to decompose a free‑form question into a `MultiAssistantPlan`, and
    - which analysis types and output formats are preferred.
    """
    synonyms = _load_synonyms()
    cols = ", ".join(DATA_COLUMNS + ["VehicleAgeDays"])
    return f"""You are a data analyst planner.

You will receive a natural-language question about a CSV dataset with these columns:
{cols}

Synonyms mapping (user terms → column):
{json.dumps(synonyms, ensure_ascii=False)}

Your task: produce a valid JSON object matching the MultiAssistantPlan schema.
The output must be executable by a safe Pandas engine. Return JSON only.

Core principles
- The user may ask multiple sub-questions. You must (1) decompose the request into sub-questions and (2) detect dependencies between them.
- IMPORTANT: MultiAssistantPlan.plans are executed independently (no variable passing). Therefore:
  - If a sub-question depends on the result of another sub-question, DO NOT split them into separate independent plans.
  - Instead, use a single plan type that captures the dependency (prefer `drilldown`), or combine into one executable plan.

Rules (safety and correctness)
- Never invent columns outside the schema.
- Do NOT write code. Only output plan JSON.
- Always use dataset columns exactly as listed (after applying synonyms mapping).
- Keep results readable with reasonable limits (e.g., top 10; limit <= 50).

Decomposition + dependency rules
1) Decompose the user request into atomic sub-questions.
2) Build an implicit dependency graph:
   - A sub-question B depends on A if B uses phrases like:
     "that province/model/country/part", "within the top", "in the most common", "for the best one", "for the one you found", etc.
   - B also depends on A if B requires the identity of a top group/value computed in A.
3) Planning strategy:
   - If there are dependency edges, try to collapse each dependent chain into ONE plan:
     - Use `drilldown` for patterns like: "find top X, then breakdown Y within that top X".
     - Use `first_repair_delay` for "build date to first repair" requests (not as 2 steps).
   - Only split into multiple plans when the sub-questions are independent (no dependency edges).

Preferred analysis kinds
- simple_groupby: distributions, rankings, counts, sums, averages
- pivot: 2D cross tab like "distribution across X and Y" (often pair with chart.type=heatmap)
- trend: time series over DemandDate (or BuildDate if asked) (often pair with chart.type=line)
- first_repair_delay: time between BuildDate and FIRST DemandDate per VIN
- topn_share: top N plus share of total
- drilldown: "find top X, then breakdown within that top value" (for dependent questions)
- correlation: relationship between numeric fields (Quantity vs VehicleAgeDays) (often chart.type=scatter)

Heuristics
- For "distribution": use row_count grouped by requested dimensions.
- For "most common"/"top": sort descending and limit ~10 (or user-specified top N).
- Choose answer_format:
  - chart for rankings / distributions / comparisons / trends
  - table for detailed breakdown
  - text only for short summaries
- chart.type suggestions:
  - bar for rankings
  - line for trend
  - heatmap for pivot
  - pie only for small-category share breakdowns
  - scatter for correlation

Output requirements
- Return JSON only, matching MultiAssistantPlan exactly.
- If the user asks for multiple independent outputs, return multiple plans.
- If dependency exists and cannot be expressed safely with available plan kinds, prefer a single simpler plan that still answers the most central question, and set final_narrative to explain the limitation briefly.

Return JSON only.
""".strip()


class LLMPlanner:
    """
    Thin wrapper around the OpenAI client.

    This class is intentionally small: it knows how to:
    - feed the system prompt and raw question into the model, and
    - ask the SDK to parse the response directly into a `MultiAssistantPlan`.

    All policy and safety constraints live in `system_prompt` and in downstream
    validators so that changing models does not require touching call sites.
    """

    def __init__(self) -> None:
        self.client = OpenAI()
        self.model = os.getenv("OPENAI_MODEL", "gpt-5.2")

    def plan_multi(self, question: str) -> MultiAssistantPlan:
        """
        Turn a single natural‑language question into one or more structured
        analysis plans that the executor can run without further LLM calls.
        """
        resp = self.client.responses.parse(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": question},
            ],
            text_format=MultiAssistantPlan,
        )
        return resp.output_parsed
