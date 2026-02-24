from __future__ import annotations

from typing import List

SUGGESTIONS = [
    "Try specifying a dimension to group by, e.g., 'by VehicleModel' or 'by Province'.",
    "If you want a trend, ask: 'Show the monthly trend of total Quantity requested.'",
    "For a 2D distribution, ask: 'Heatmap of DemandType by VehicleModel using row counts.'",
    "To drill down, ask: 'Which province has the most requests, and within that province what is the PayType distribution?'",
    "If the result is too large, ask for 'top 10' or add filters such as CountryCode or VehicleModel.",
]

def default_suggestions() -> List[str]:
    return SUGGESTIONS[:]
