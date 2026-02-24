from __future__ import annotations

from typing import List

# Curated prompts that gently steer the user towards questions that map well
# onto the planning/execution capabilities of this agent.
SUGGESTIONS = [
    "Try specifying a dimension to group by, e.g., 'by VehicleModel' or 'by Province'.",
    "If you want a trend, ask: 'Show the monthly trend of total Quantity requested.'",
    "For a 2D distribution, ask: 'Heatmap of DemandType by VehicleModel using row counts.'",
    "To drill down, ask: 'Which province has the most requests, and within that province what is the PayType distribution?'",
    "If the result is too large, ask for 'top 10' or add filters such as CountryCode or VehicleModel.",
]


def default_suggestions() -> List[str]:
    """Return a shallow copy so callers cannot accidentally mutate the global."""
    return SUGGESTIONS[:]
