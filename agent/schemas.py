from enum import StrEnum

from pydantic import BaseModel, Field


class ResponseMode(StrEnum):
    RAW_LIST = "raw_list"
    LIST_WITH_INSIGHTS = "list_with_insights"
    REPORT = "report"


class RoutingTaskType(StrEnum):
    LOOKUP = "lookup"
    SEARCH = "search"
    REPORT = "report"
    LOOKUP_THEN_REPORT = "lookup_then_report"


class RoutingDecision(BaseModel):
    task_type: RoutingTaskType = Field(description="The user's high-level task type.")
    response_mode: ResponseMode = Field(description="How the final answer should be rendered.")
    search_mode: str = Field(description="One of metadata, semantic, hybrid.")
    user_goal: str = Field(description="A short description of what the user wants.")
    query_text: str | None = Field(default=None, description="Core semantic query without metadata constraints.")
    title: str | None = Field(default=None, description="Paper title or title fragment for precise lookup.")
    authors: str | None = Field(default=None, description="Author names, comma-separated if multiple.")
    categories: str | None = Field(default=None, description="arXiv categories, comma-separated if multiple.")
    comment: str | None = Field(default=None, description="Venue, journal, or comment keywords.")
    published: str | None = Field(
        default=None,
        description="Published time filter such as 2024, after:2024, since:2024-01-01, between:2023-01-01,2024-12-31, recent:2y.",
    )
    top_k: int = Field(default=20, ge=1, le=50, description="How many papers to retrieve.")
