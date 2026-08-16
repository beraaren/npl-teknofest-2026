from typing import Optional, Dict, Any, Callable
from fastapi import APIRouter
from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    service: str
    dlq_count: int = 0
    details: Optional[Dict[str, Any]] = None

def create_health_router(
    service_name: str,
    get_details_fn: Optional[Callable[[], Dict[str, Any]]] = None
) -> APIRouter:
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    async def health_check():
        details = get_details_fn() if get_details_fn else {}
        dlq_cnt = details.get("dlq_count", 0) if isinstance(details, dict) else 0
        return HealthResponse(
            status="ok",
            service=service_name,
            dlq_count=dlq_cnt,
            details=details if details else None
        )

    return router
