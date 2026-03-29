from fastapi import APIRouter

from agrigee_lite.api.routes.images import router as images_router
from agrigee_lite.api.routes.jobs import router as jobs_router
from agrigee_lite.api.routes.sits import router as sits_router

router = APIRouter()
router.include_router(images_router)
router.include_router(sits_router)
router.include_router(jobs_router)

__all__ = ["router"]
