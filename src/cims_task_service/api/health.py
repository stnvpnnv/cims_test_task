"""Operational health endpoints."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    """Liveness response returned by a running API process."""

    status: Literal["ok"] = "ok"


@router.get(
    "/live",
    response_model=HealthResponse,
    summary="Check API process liveness",
)
async def check_liveness() -> HealthResponse:
    """Report that the API process can serve requests."""

    return HealthResponse()
