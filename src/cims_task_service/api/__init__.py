"""HTTP API router composition."""

from fastapi import APIRouter

from cims_task_service.api.health import router as health_router

router = APIRouter()
router.include_router(health_router)

__all__ = ["router"]
