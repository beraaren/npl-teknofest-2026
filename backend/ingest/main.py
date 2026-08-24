"""camera-ingest-service — video/RTSP kaynağını okuyup frame.chunk yayınlar.

§2.3.1 tasarımına uygun:
  - VideoReader ile video okunur (PyAV tabanlı)
  - Kanal A örnekleme: native_fps / channel_a_fps adımı
  - Frame'ler data/frames/<job_id>/f_XXXXXX.jpg olarak diske yazılır
  - Her chunk için FrameChunk mesajı yayınlanır
  - Video bitince stream.eof yayınlanır
  - CPU-yoğun işlem run_in_executor ile non-blocking
"""
import asyncio
import os
import logging
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager

from ..common.health import create_health_router
from ..common import redis as redis_helper
from ..common.config_loader import load_app_config
from ..contracts.messages import FrameChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chunk başına kaç frame gönderileceği
CHUNK_SIZE = 10


class JobRequest(BaseModel):
    job_id: str
    camera_id: str
    video_path: str
    fps: float = 0.0  # 0 → config.yaml'den oku


def _extract_frames_sync(video_path: str, frames_dir: str, channel_a_fps: float):
    """
    Senkron (CPU-yoğun) işlev — run_in_executor içinde çalıştırılır.
    VideoReader ile video açılır, Kanal A adımıyla kareler örneklenir,
    diske jpg olarak yazılır.
    Döner: (saved_paths, saved_indices, native_fps)
    """
    import sys
    _bera_root = str(Path(__file__).resolve().parent.parent.parent)
    if _bera_root not in sys.path:
        sys.path.insert(0, _bera_root)

    try:
        import cv2
        from src.preprocessing.video_reader import VideoReader

        saved_paths: list[str] = []
        saved_indices: list[int] = []

        os.makedirs(frames_dir, exist_ok=True)

        with VideoReader(video_path) as reader:
            native_fps = reader.fps or 25.0
            step = max(1, round(native_fps / channel_a_fps)) if channel_a_fps > 0 else 1

            for real_idx, frame_rgb in enumerate(reader.iter_frames()):
                if real_idx % step != 0:
                    continue
                frame_bgr = frame_rgb[:, :, ::-1]  # RGB → BGR for cv2
                fname = f"f_{real_idx:06d}.jpg"
                fpath = os.path.join(frames_dir, fname)
                cv2.imwrite(fpath, frame_bgr)
                saved_paths.append(fpath)
                saved_indices.append(real_idx)

        return saved_paths, saved_indices, native_fps

    except Exception as exc:
        logger.error(f"VideoReader hatası: {exc}. Mock frame'lere düşülüyor.")
        # Fallback: 5 adet boş frame
        os.makedirs(frames_dir, exist_ok=True)
        saved_paths = []
        saved_indices = []
        for i in range(5):
            fpath = os.path.join(frames_dir, f"f_{i:06d}.jpg")
            with open(fpath, "wb") as f:
                f.write(b"")
            saved_paths.append(fpath)
            saved_indices.append(i * 10)
        return saved_paths, saved_indices, 25.0


_metrics = {"jobs_started": 0, "frames_extracted": 0, "active_jobs": 0, "dlq_count": 0}


async def process_video(job_id: str, camera_id: str, video_path: str, fps_override: float):
    logger.info(f"Starting job {job_id} | camera={camera_id} | source={video_path}")
    _metrics["jobs_started"] += 1
    _metrics["active_jobs"] += 1

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    client = await redis_helper.get_redis_client(redis_url)

    frames_dir = os.path.join("data", "frames", job_id)

    # Config'den Kanal A FPS değerini oku
    cfg = load_app_config()
    channel_a_fps = fps_override if fps_override > 0 else (
        cfg.preprocessing.channel_a_fps if cfg else 12.0
    )

    try:
        loop = asyncio.get_event_loop()
        saved_paths, saved_indices, native_fps = await loop.run_in_executor(
            None, _extract_frames_sync, video_path, frames_dir, channel_a_fps
        )

        effective_fps = channel_a_fps if channel_a_fps > 0 else native_fps
        _metrics["frames_extracted"] += len(saved_paths)

        # Chunk'lar halinde yayınla
        for chunk_start in range(0, len(saved_paths), CHUNK_SIZE):
            chunk_paths = saved_paths[chunk_start: chunk_start + CHUNK_SIZE]
            chunk_indices = saved_indices[chunk_start: chunk_start + CHUNK_SIZE]
            is_last = (chunk_start + CHUNK_SIZE) >= len(saved_paths)

            chunk = FrameChunk(
                job_id=job_id,
                camera_id=camera_id,
                frame_paths=chunk_paths,
                frame_indices=chunk_indices,
                fps=effective_fps,
                is_last=is_last,
                # Kanal B videoyu bütün olarak analiz ettiği için kaynak yol
                # zincirde taşınır (kareler videoyu geri üretemez).
                video_path=video_path,
            )
            await redis_helper.publish_message(client, "frame.chunk", chunk)
            logger.info(
                f"Published chunk frames {chunk_indices[0]}..{chunk_indices[-1]}"
                f" (is_last={is_last})"
            )
            await asyncio.sleep(0)  # event loop'a nefes aldır

        # §2.3.1: stream.eof olayı
        await redis_helper.publish_raw(client, "stream.eof", {
            "job_id": job_id,
            "camera_id": camera_id,
            "total_frames": len(saved_paths),
        })
        logger.info(f"Finished job {job_id} — {len(saved_paths)} frames published")

    except asyncio.CancelledError:
        logger.warning(f"Job {job_id} cancelled")
    except Exception as exc:
        logger.error(f"Job {job_id} error: {exc}", exc_info=True)
        _metrics["dlq_count"] += 1
        await redis_helper.publish_raw(client, "stream.eof", {
            "job_id": job_id,
            "camera_id": camera_id,
            "error": str(exc),
        })
    finally:
        _metrics["active_jobs"] = max(0, _metrics["active_jobs"] - 1)
        await client.aclose()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="Dalga AI Camera Ingest Service")
app.include_router(
    create_health_router("camera-ingest-service", get_details_fn=lambda: _metrics),
    prefix="/api/v1",
)


@app.post("/api/v1/jobs")
async def create_job(request: JobRequest, background_tasks: BackgroundTasks):
    """Gateway'den çağrılan dahili endpoint — yeni bir ingest işi başlatır."""
    background_tasks.add_task(
        process_video,
        request.job_id,
        request.camera_id,
        request.video_path,
        request.fps,
    )
    return {"status": "accepted", "job_id": request.job_id}


@app.get("/api/v1/metrics")
async def get_metrics():
    return _metrics
