"""cameras router — §2.5 tasarımına uygun.

DELETE /cameras/{id}: DB kaydını siler + ingest.stop komutu yayınlar.
"""
import os
import logging
import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from .. import store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cameras", tags=["cameras"])


class CameraCreate(BaseModel):
    name: str
    source: str


class CameraStatusUpdate(BaseModel):
    status: str


@router.post("/", response_model=dict)
def add_camera(camera: CameraCreate):
    return store.create_camera(camera.name, camera.source)


@router.get("/", response_model=List[dict])
def list_cameras():
    return store.get_cameras()


@router.get("/{cam_id}", response_model=dict)
def get_camera_by_id(cam_id: str):
    cam = store.get_camera(cam_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Kamera bulunamadı")
    return cam


@router.patch("/{cam_id}/status", response_model=dict)
def update_camera_status_endpoint(cam_id: str, body: CameraStatusUpdate):
    cam = store.get_camera(cam_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Kamera bulunamadı")
    store.update_camera_status(cam_id, body.status)
    return {"status": "updated", "id": cam_id, "camera_status": body.status}


@router.delete("/{cam_id}")
async def remove_camera(cam_id: str):
    """
    §2.5: Kamerayı siler ve aktif ingest işini durdurur.
    Redis ingest.stop stream'ine stop komutu yayınlanır.
    """
    from ...common import redis as redis_helper

    # DB kaydını sil
    store.delete_camera(cam_id)

    # Aktif ingest'i durdurma sinyali gönder
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    try:
        client = await redis_helper.get_redis_client(redis_url)
        await client.xadd("ingest.stop", {"camera_id": cam_id})
        await client.aclose()
        logger.info(f"Kamera {cam_id} silindi ve ingest.stop sinyali gönderildi.")
    except Exception as exc:
        logger.warning(f"ingest.stop sinyali gönderilemedi: {exc}")

    return {"status": "deleted", "id": cam_id}
