"""Problem Details response schemas for semantic API errors."""

from pydantic import BaseModel, ConfigDict, Field


class ProblemDetails(BaseModel):
    """RFC 9457-compatible details for an unsuccessful HTTP operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = Field(description="Stable URI reference identifying the problem type.")
    title: str = Field(description="Short, human-readable summary of the problem type.")
    status: int = Field(ge=400, le=599, description="HTTP status code for this occurrence.")
    detail: str = Field(description="Human-readable explanation of this occurrence.")
