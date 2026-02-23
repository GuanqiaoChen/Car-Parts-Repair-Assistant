from __future__ import annotations

from typing import Literal, Optional, List, Union
from pydantic import BaseModel, Field, ConfigDict

# Dataset columns (per assignment)
DATA_COLUMNS = [
    "DemandDate",
    "BuildDate",
    "Province",
    "CountryCode",
    "Quantity",
    "SSPart",
    "RequestType",
    "DemandType",
    "PayType",
    "VehicleModel",
    "VIN",
]

# Supported operators / aggregations
Agg = Literal["count", "sum", "mean", "min", "max", "nunique", "median"]
Op = Literal["==", "!=", "in", "contains", ">=", "<=", "between"]
ChartType = Literal["none", "bar", "line", "pie", "heatmap"]
AnswerFormat = Literal["text", "table", "chart"]

Dim = Literal["Province", "CountryCode", "SSPart", "RequestType", "DemandType", "PayType", "VehicleModel", "VIN"]
DateCol = Literal["DemandDate", "BuildDate"]
MetricCol = Literal["Quantity", "VehicleAgeDays", "delay_days", "row_count"]

class FilterSpec(BaseModel):
    """A safe, whitelisted filter specification."""
    model_config = ConfigDict(extra="forbid")

    col: Literal[
        "DemandDate", "BuildDate",
        "Province", "CountryCode", "Quantity",
        "SSPart", "RequestType", "DemandType", "PayType", "VehicleModel", "VIN",
        "VehicleAgeDays",
    ]
    op: Op
    value: Union[str, int, float, List[Union[str, int, float]]]

class MetricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    col: Literal["Quantity", "VehicleAgeDays", "row_count"]
    agg: Agg
    alias: str = "metric"

class SortSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    by: str
    ascending: bool = False

class ChartSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: ChartType = "none"
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    title: Optional[str] = None

class SimpleGroupbyPlan(BaseModel):
    """General groupby plan: counts/sums/means by one or more dimensions."""
    model_config = ConfigDict(extra="forbid")

    kind: Literal["simple_groupby"] = "simple_groupby"
    filters: List[FilterSpec] = Field(default_factory=list)
    groupby: List[Dim] = Field(default_factory=list)
    metrics: List[MetricSpec] = Field(default_factory=list)
    sort: Optional[SortSpec] = None
    limit: int = 50

class PivotPlan(BaseModel):
    """2D pivot for heatmaps / cross-tab distributions."""
    model_config = ConfigDict(extra="forbid")

    kind: Literal["pivot"] = "pivot"
    filters: List[FilterSpec] = Field(default_factory=list)
    rows: Dim
    cols: Dim
    value: Literal["Quantity", "row_count"] = "row_count"
    agg: Literal["sum", "count"] = "count"
    limit_rows: int = 25
    limit_cols: int = 25

class TrendPlan(BaseModel):
    """Time series trend on DemandDate (or BuildDate)."""
    model_config = ConfigDict(extra="forbid")

    kind: Literal["trend"] = "trend"
    filters: List[FilterSpec] = Field(default_factory=list)
    date_col: DateCol = "DemandDate"
    freq: Literal["D", "W", "M"] = "M"   # day / week / month
    groupby: Optional[Dim] = None       # optional series split (e.g., by VehicleModel)
    metric: Literal["Quantity", "row_count"] = "row_count"
    agg: Literal["sum", "count"] = "count"
    limit_series: int = 10
    limit_points: int = 200

class FirstRepairDelayPlan(BaseModel):
    """BuildDate -> first DemandDate per VIN; aggregate delay_days."""
    model_config = ConfigDict(extra="forbid")

    kind: Literal["first_repair_delay"] = "first_repair_delay"
    filters: List[FilterSpec] = Field(default_factory=list)
    by: Literal["VehicleModel", "Province", "CountryCode"] = "VehicleModel"
    agg: Literal["mean", "median", "min", "max"] = "mean"
    sort_desc: bool = False
    limit: int = 50

AnalysisPlan = Union[SimpleGroupbyPlan, PivotPlan, TrendPlan, FirstRepairDelayPlan]

class AssistantPlan(BaseModel):
    """Top-level plan returned by the LLM."""
    model_config = ConfigDict(extra="forbid")

    answer_format: AnswerFormat = "table"
    analysis: AnalysisPlan
    chart: ChartSpec = Field(default_factory=ChartSpec)
    narrative: str = Field(default="", description="One short sentence describing what will be computed.")

class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str

class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_format: AnswerFormat
    narrative: str
    text: Optional[str] = None
    table: Optional[dict] = None   # {"columns": [...], "rows": [[...], ...]}
    chart: Optional[dict] = None   # Plotly figure JSON
    plan: dict
