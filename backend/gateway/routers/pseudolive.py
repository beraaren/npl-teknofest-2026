"""Pseudolive replay router — kamera listesi ve aktif video servisi."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from ..replay import ReplayEngine

router = APIRouter(prefix="/pseudolive", tags=["pseudolive"])


def _get_engine(request: Request) -> ReplayEngine:
    return request.app.state.replay_engine


@router.get("/cameras")
async def list_pseudolive_cameras(request: Request):
    """9 pseudolive kameranın aktif analiz özetini döner."""
    engine = _get_engine(request)
    cameras = []
    for camera_id in [f"cam-{i:02d}" for i in range(1, 10)]:
        active = engine.get_camera_status(camera_id)
        cameras.append({
            "camera_id": camera_id,
            "active": active,
        })
    return cameras


@router.get("/videos/{camera_id}")
async def get_video(camera_id: str, request: Request):
    """Aktif videoyu HTTP Range desteğiyle sunar."""
    engine = _get_engine(request)
    video_path = engine.current_video_path(camera_id)
    if not video_path or not video_path.exists():
        raise HTTPException(status_code=404, detail="Aktif video bulunamadı")
    return range_file_response(video_path, request)


def _parse_range_header(range_header: str, file_size: int) -> tuple[int, int] | None:
    """bytes=start-end formatını ayrıştırır; geçersizse None döner."""
    if not range_header.startswith("bytes="):
        return None
    try:
        ranges = range_header[len("bytes="):].strip().split(",")
        spec = ranges[0].strip()
        if "-" not in spec:
            return None
        start_str, end_str = spec.split("-", 1)
        if start_str == "" and end_str != "":
            # Son N byte: bytes=-500
            suffix = int(end_str)
            if suffix <= 0:
                return None
            end = file_size - 1
            start = max(0, file_size - suffix)
        elif start_str != "":
            start = int(start_str)
            if end_str == "":
                end = file_size - 1
            else:
                end = int(end_str)
            if start < 0 or start >= file_size or end < start:
                return None
            end = min(end, file_size - 1)
        else:
            return None
        return start, end
    except ValueError:
        return None


def range_file_response(file_path: Path, request: Request):
    """Range header'ına göre 206 Partial Content veya 200 Full Content döner."""
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")

    if not range_header:
        return FileResponse(
            file_path,
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes"},
        )

    parsed = _parse_range_header(range_header, file_size)
    if parsed is None:
        # Geçersiz range isteği: tüm dosyayı 200 ile döndür
        return FileResponse(
            file_path,
            media_type="video/mp4",
            headers={"Accept-Ranges": "bytes"},
        )

    start, end = parsed
    length = end - start + 1

    def iter_file():
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = length
            chunk_size = 64 * 1024
            while remaining > 0:
                to_read = min(chunk_size, remaining)
                data = f.read(to_read)
                if not data:
                    break
                yield data
                remaining -= len(data)

    return StreamingResponse(
        iter_file(),
        status_code=206,
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(length),
        },
    )
