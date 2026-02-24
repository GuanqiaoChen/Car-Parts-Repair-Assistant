from __future__ import annotations

from typing import Literal, Optional, List, Union
from pydantic import BaseModel, Field, ConfigDict

# Columns exposed to the planning layer; these mirror what 'data.load_dataset'
# guarantees after preprocessing. Keeping them in one place makes it easier to
# reason about what the LLM is allowed to reference.
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
    """
    Declarative description of a single filter clause that can safely be
    applied to the Pandas DataFrame.
    """

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
    """
    Describes a single aggregation to compute, such as 'sum(Quantity)' or a
    synthetic 'row_count'. The alias is what eventually surfaces in tables.
    """

    model_config = ConfigDict(extra="forbid")
    col: Literal["Quantity", "VehicleAgeDays", "row_count"]
    agg: Agg
    alias: str = "metric"


class SortSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    by: str
    ascending: bool = False


class ChartSpec(BaseModel):
    """
    Thin description of how a result should be plotted. The executor is free
    to ignore this when a chart is not feasible.
    """

    model_config = ConfigDict(extra="forbid")
    type: ChartType = "none"
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    title: Optional[str] = None


class SimpleGroupbyPlan(BaseModel):
    """
    Generic "group by dimensions and aggregate metrics" building block.
    Covers distributions, rankings, and many slice-and-dice questions.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["simple_groupby"] = "simple_groupby"
    filters: List[FilterSpec] = Field(default_factory=list)
    groupby: List[Dim] = Field(default_factory=list)
    metrics: List[MetricSpec] = Field(default_factory=list)
    sort: Optional[SortSpec] = None
    limit: int = 50


class PivotPlan(BaseModel):
    """
    2D cross-tabulation for questions like "distribution across X and Y".
    """

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
    """
    Time-bucketed series over 'DemandDate' or 'BuildDate', optionally split
    into multiple series by a dimension such as 'VehicleModel'.
    """

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
    """
    Specialised plan for "build date to first repair" style questions.
    """

    model_config = ConfigDict(extra="forbid")
    kind: Literal["first_repair_delay"] = "first_repair_delay"
    filters: List[FilterSpec] = Field(default_factory=list)
    by: Literal["VehicleModel", "Province", "CountryCode"] = "VehicleModel"
    agg: Literal["mean", "median", "min", "max"] = "mean"
    sort_desc: bool = False
    limit: int = 50


class TopNSharePlan(BaseModel):
    """
    Top-N breakdown plus share of total, for "what are the leading groups?"
    type questions.
    """

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
    """
    Two-stage plan: first identify the top group by 'top_dim', then compute a
    breakdown within that winning group along 'breakdown_dim'.
    """

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
    """
    Compute correlation between two numeric fields, optionally on a sample
    for performance reasons. Often paired with a scatter plot.
    """

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
    """
    One executable step in the multi-step plan the LLM returns.

    It ties together:
    - the core analysis description,
    - an optional visualisation hint, and
    - a short narrative that explains the intent in plain language.
    """

    model_config = ConfigDict(extra="forbid")
    answer_format: AnswerFormat = "table"
    analysis: AnalysisPlan
    chart: ChartSpec = Field(default_factory=ChartSpec)
    narrative: str = Field(default="", description="One short sentence describing what will be computed.")


class MultiAssistantPlan(BaseModel):
    """
    Container for a small number of independent (or loosely related) steps
    that together answer the user's question.
    """

    model_config = ConfigDict(extra="forbid")
    plans: List[AssistantPlan]
    final_answer_format: AnswerFormat = "table"
    final_narrative: str = ""


class QueryRequest(BaseModel):
    """Shape of the JSON payload sent from the Streamlit app to the API."""

    model_config = ConfigDict(extra="forbid")
    question: str


class ResultItem(BaseModel):
    """
    One rendered step in the response: narrative + explanation plus either
    text, a table, a chart, or some combination of those.
    """

    model_config = ConfigDict(extra="forbid")

    # 'ok'      : the step executed normally.
    # 'preview' : the backend applied a system-level truncation and is returning
    #             a limited preview instead of the full output.
    # 'error'   : the step could not be executed as requested.
    status: Literal["ok", "preview", "error"] = "ok"
    notices: List[str] = Field(default_factory=list)

    answer_format: AnswerFormat
    narrative: str
    explanation: str
    text: Optional[str] = None
    table: Optional[dict] = None
    chart: Optional[dict] = None
    plan: dict

class QueryResponse(BaseModel):
    """
    Full response contract used by the frontend, including per-step details,
    a final narrative, and optional follow-up question suggestions.
    """

    model_config = ConfigDict(extra="forbid")
    items: List[ResultItem]
    final_narrative: str
    suggestions: List[str] = Field(default_factory=list)
