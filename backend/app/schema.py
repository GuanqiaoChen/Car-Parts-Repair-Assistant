from __future__ import annotations

from typing import Literal, Optional, List, Union
from pydantic import BaseModel, Field, ConfigDict

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

Agg = Literal["count", "sum", "mean", "min", "max", "nunique", "median"]
Op = Literal["==", "!=", "in", "contains", ">=", "<=", "between"]
ChartType = Literal["none", "bar", "line", "pie", "heatmap", "scatter"]
AnswerFormat = Literal["text", "table", "chart"]

Dim = Literal["Province", "CountryCode", "SSPart", "RequestType", "DemandType", "PayType", "VehicleModel", "VIN"]
DateCol = Literal["DemandDate", "BuildDate"]

class FilterSpec(BaseModel):
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
    model_config = ConfigDict(extra="forbid")
    kind: Literal["simple_groupby"] = "simple_groupby"
    filters: List[FilterSpec] = Field(default_factory=list)
    groupby: List[Dim] = Field(default_factory=list)
    metrics: List[MetricSpec] = Field(default_factory=list)
    sort: Optional[SortSpec] = None
    limit: int = 50

class PivotPlan(BaseModel):
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
    model_config = ConfigDict(extra="forbid")
    kind: Literal["trend"] = "trend"
    filters: List[FilterSpec] = Field(default_factory=list)
    date_col: DateCol = "DemandDate"
    freq: Literal["D", "W", "M"] = "M"
    groupby: Optional[Dim] = None
    metric: Literal["Quantity", "row_count"] = "row_count"
    agg: Literal["sum", "count"] = "count"
    limit_series: int = 10
    limit_points: int = 200

class FirstRepairDelayPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["first_repair_delay"] = "first_repair_delay"
    filters: List[FilterSpec] = Field(default_factory=list)
    by: Literal["VehicleModel", "Province", "CountryCode"] = "VehicleModel"
    agg: Literal["mean", "median", "min", "max"] = "mean"
    sort_desc: bool = False
    limit: int = 50

class TopNSharePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["topn_share"] = "topn_share"
    filters: List[FilterSpec] = Field(default_factory=list)
    dim: Dim
    metric: Literal["Quantity", "row_count"] = "row_count"
    agg: Literal["sum", "count"] = "count"
    top_n: int = 10
    include_other: bool = True
    sort_desc: bool = True

class DrilldownPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["drilldown"] = "drilldown"
    filters: List[FilterSpec] = Field(default_factory=list)
    top_dim: Dim
    top_metric: Literal["Quantity", "row_count"] = "row_count"
    top_agg: Literal["sum", "count"] = "count"
    top_n: int = 1
    breakdown_dim: Dim
    breakdown_metric: Literal["Quantity", "row_count"] = "row_count"
    breakdown_agg: Literal["sum", "count"] = "count"
    breakdown_limit: int = 20

class CorrelationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["correlation"] = "correlation"
    filters: List[FilterSpec] = Field(default_factory=list)
    x: Literal["Quantity", "VehicleAgeDays"]
    y: Literal["Quantity", "VehicleAgeDays"]
    method: Literal["pearson", "spearman"] = "pearson"
    sample: int = 5000

AnalysisPlan = Union[
    SimpleGroupbyPlan,
    PivotPlan,
    TrendPlan,
    FirstRepairDelayPlan,
    TopNSharePlan,
    DrilldownPlan,
    CorrelationPlan,
]

class AssistantPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer_format: AnswerFormat = "table"
    analysis: AnalysisPlan
    chart: ChartSpec = Field(default_factory=ChartSpec)
    narrative: str = Field(default="", description="One short sentence describing what will be computed.")

class MultiAssistantPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plans: List[AssistantPlan]
    final_answer_format: AnswerFormat = "table"
    final_narrative: str = ""

class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str

class ResultItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer_format: AnswerFormat
    narrative: str
    explanation: str
    text: Optional[str] = None
    table: Optional[dict] = None
    chart: Optional[dict] = None
    plan: dict

class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: List[ResultItem]
    final_narrative: str
    suggestions: List[str] = Field(default_factory=list)
