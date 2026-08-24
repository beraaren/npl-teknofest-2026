"""Sözde-canlı kamera duvarı ve analiz kütüphanesi uçları.

Kamera duvarı, kaydedilmiş analizleri canlı gibi oynatır (bkz. ``replay.py``).
Bu router iki şey sunar:

* **Kamera uçları** (``/pseudolive/...``): duvardaki kameraların anlık durumu ve
  o an oynayan video. Arayüz, kamera durumundaki ``position_sec`` değerini
  kullanarak video öğesini sunucunun sanal zamanına hizalar.
* **Kütüphane uçları** (``/library/...``): analizlere slug ile erişim. Atama
  gören saha ekibi, kamerada ne oynadığından bağımsız olarak kendi olayının
  videosunu buradan alır.

Video dosyaları neden slug ile sunuluyor: kütüphanedeki dosya adları emoji,
Latin dışı harfler, boşluk ve parantez içeriyor. Bu adları URL yolunda taşımak
kırılgandır; arayüz kararlı bir slug kullanır, gerçek dosya yolu sunucuda
çözülür.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

from .. import library as catalog
from ..replay import ReplayEngine

router = APIRouter(tags=["pseudolive"])


def _get_engine(request: Request) -> ReplayEngine:
    return request.app.state.replay_engine


# ---------------------------------------------------------------------------
# Kamera duvarı
# ---------------------------------------------------------------------------

@router.get("/pseudolive/cameras")
async def list_pseudolive_cameras(request: Request):
    """Duvardaki kameraların anlık durumunu döner.

    Her kayıt, o an oynayan analizin özetini ve sanal oynatma konumunu
    (``position_sec``) taşır; arayüz videoyu bu konuma çeker.
    """
    engine = _get_engine(request)
    return [
        {"camera_id": camera_id, "active": engine.get_camera_status(camera_id)}
        for camera_id in engine.cameras
    ]


@router.get("/pseudolive/cameras/{camera_id}")
async def get_pseudolive_camera(camera_id: str, request: Request):
    """Tek bir kameranın durumunu ve oynayan analizin tamamını döner."""
    engine = _get_engine(request)
    status = engine.get_camera_status(camera_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Kamera bulunamadı: {camera_id}")

    analysis = engine.current_analysis(camera_id)
    return {
        "camera_id": camera_id,
        "active": status,
        "analysis": catalog.public_view(analysis) if analysis else None,
    }


@router.get("/pseudolive/videos/{camera_id}")
async def get_camera_video(camera_id: str, request: Request):
    """Kamerada o an oynayan videoyu Range desteğiyle sunar."""
    engine = _get_engine(request)
    video_path = engine.current_video_path(camera_id)
    if not video_path or not video_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"'{camera_id}' için aktif video bulunamadı. "
                   f"Kütüphanede {catalog.count()} analiz var.",
        )
    return range_file_response(video_path, request)


# ---------------------------------------------------------------------------
# Analiz kütüphanesi
# ---------------------------------------------------------------------------

@router.get("/library/analyses")
async def list_library_analyses(
    risk: str | None = Query(default=None, description="Risk seviyesine göre filtre"),
):
    """Kütüphanedeki tüm analizleri (gösterim biçiminde) döner."""
    items = [catalog.public_view(a) for a in catalog.all_analyses()]
    if risk:
        items = [i for i in items if str(i.get("risk")) == risk]
    return items


@router.get("/library/analyses/{slug}")
async def get_library_analysis(slug: str):
    """Tek bir analizi döner.

    Raises:
        HTTPException: Analiz yoksa 404.
    """
    analysis = catalog.get(slug)
    if analysis is None:
        raise HTTPException(status_code=404, detail=f"Analiz bulunamadı: {slug}")
    return catalog.public_view(analysis)


@router.post("/library/reload")
async def reload_library():
    """Kütüphaneyi diskten yeniden okur (yeni analiz üretildiğinde kullanılır)."""
    return {"count": catalog.reload()}


@router.get("/library/videos/{slug}")
async def get_library_video(slug: str, request: Request):
    """Belirli bir analizin videosunu Range desteğiyle sunar.

    Saha ekibi ekranı, atanan olayın videosunu bu uçtan alır ve olay anına
    konumlanır.

    Raises:
        HTTPException: Analiz veya dosya yoksa 404.
    """
    video_path = catalog.video_path(slug)
    if not video_path:
        raise HTTPException(status_code=404, detail=f"Video bulunamadı: {slug}")
    return range_file_response(video_path, request)


# ---------------------------------------------------------------------------
# HTTP Range desteği
# ---------------------------------------------------------------------------

def _parse_range_header(range_header: str, file_size: int) -> tuple[int, int] | None:
    """``bytes=start-end`` biçimini ayrıştırır; geçersizse ``None`` döner."""
    if not range_header.startswith("bytes="):
        return None
    try:
        spec = range_header[len("bytes="):].strip().split(",")[0].strip()
        if "-" not in spec:
            return None
        start_str, end_str = spec.split("-", 1)
        if start_str == "" and end_str != "":
            # Son N byte: bytes=-500
            suffix = int(end_str)
            if suffix <= 0:
                return None
            return max(0, file_size - suffix), file_size - 1
        if start_str == "":
            return None
        start = int(start_str)
        if start < 0 or start >= file_size:
            return None
        end = file_size - 1 if end_str == "" else min(int(end_str), file_size - 1)
        if end < start:
            return None
        return start, end
    except ValueError:
        return None


def range_file_response(file_path: Path, request: Request):
    """Range başlığına göre 206 Partial Content veya 200 tam içerik döner.

    Video öğesinin ileri/geri sarabilmesi ve belirli bir saniyeye
    konumlanabilmesi için Range desteği zorunludur.
    """
    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")
    full_headers = {"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"}

    if not range_header:
        return FileResponse(file_path, media_type="video/mp4", headers=full_headers)

    parsed = _parse_range_header(range_header, file_size)
    if parsed is None:
        return FileResponse(file_path, media_type="video/mp4", headers=full_headers)

    start, end = parsed
    length = end - start + 1

    def iter_file():
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = length
            chunk_size = 64 * 1024
            while remaining > 0:
                data = f.read(min(chunk_size, remaining))
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
            "Cache-Control": "public, max-age=3600",
        },
    )
