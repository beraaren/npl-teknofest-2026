import uuid
import logging
import os
import httpx
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from .. import store

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analyses"])
INGEST_SERVICE_URL = os.environ.get("INGEST_SERVICE_URL", "http://camera-ingest:8001")


class AnalysisRequest(BaseModel):
    video_path: str
    camera_id: str = "batch_file"
    fps: float = 0.0


@router.post("/analyses", status_code=202)
async def create_analysis(request: AnalysisRequest):
    """
    §2.5: Yeni bir analiz işi başlatır.
    Gateway ingest servisini tetikler; pipeline kendi kendine akar.
    """
    job_id = str(uuid.uuid4())

    # İngest servisini tetikle (dahili çağrı)
    ingest_payload = {
        "job_id": job_id,
        "camera_id": request.camera_id,
        "video_path": request.video_path,
        "fps": request.fps,
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{INGEST_SERVICE_URL}/api/v1/jobs", json=ingest_payload)
            resp.raise_for_status()
    except Exception as exc:
        logger.error(f"Ingest servisi tetiklenemedi: {exc}")
        raise HTTPException(status_code=503, detail=f"Ingest servisi erişilemiyor: {exc}")

    return {"status": "accepted", "job_id": job_id}


@router.get("/analyses")
async def list_analyses(
    camera_id: Optional[str] = Query(default=None),
    risk: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """§2.5: Analizleri filtrele (camera_id, risk, sayfalama)."""
    return store.get_analyses(camera_id=camera_id, risk=risk, limit=limit, offset=offset)


@router.get("/analyses/{job_id}")
async def get_analysis(job_id: str):
    """§2.5: Tam analiz sonucunu döner."""
    result = store.get_analysis_by_id(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analiz bulunamadı")
    return result


@router.get("/analyses/{job_id}/events")
async def get_analysis_events(job_id: str):
    """
    §2.5: Job'a ait tüm akış olaylarını (event.detected, tool.executed,
    notification.push, decision.final) kronolojik sırayla döner.
    Demo ve debug için kullanılır.
    """
    events = store.get_events_by_job(job_id)
    if not events:
        raise HTTPException(status_code=404, detail="Bu job için olay bulunamadı")
    return {"job_id": job_id, "count": len(events), "events": events}
