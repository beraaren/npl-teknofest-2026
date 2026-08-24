"""perception-service — frame.chunk tüketip event.detected yayınlar.

§2.3.2 tasarımına uygun:
  - frame.chunk'ları job bazında tamponlar
  - is_last=True veya pencere dolunca ObserverAgent + EventEngine çalışır
  - Gerçek YOLO+ByteTrack → sahne grafi → 8 geometrik kural
  - Her sinyal için EventDetected mesajı yayınlanır
  - CPU-yoğun işlem run_in_executor ile non-blocking
"""
import asyncio
import json
import os
import logging
from collections import defaultdict
from pathlib import Path
from fastapi import FastAPI
from contextlib import asynccontextmanager

from ..common.health import create_health_router
from ..common import redis as redis_helper
from ..common.config_loader import load_app_config
from ..contracts.messages import FrameChunk, EventDetected

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrik sayaçları
_metrics = {"processed_frames": 0, "events_detected": 0, "jobs_processed": 0, "dlq_count": 0}

# Job başına frame buffer: {job_id: {"paths": [], "indices": [], "fps": float}}
_job_buffers: dict = defaultdict(lambda: {"paths": [], "indices": [], "fps": 25.0})


def _run_perception_sync(frame_paths: list[str], sampled_indices: list[int], fps: float):
    """
    Senkron CPU-yoğun işlev — run_in_executor içinde çalıştırılır.
    ObserverAgent + EventEngine çağrılır; EventSignal listesi döner.
    """
    import sys
    _bera_root = str(Path(__file__).resolve().parent.parent.parent)
    if _bera_root not in sys.path:
        sys.path.insert(0, _bera_root)

    try:
        import cv2
        from src.config import load_config
        from src.perception.observer_agent import ObserverAgent
        from src.events.event_engine import EventEngine

        cfg = load_config(
            os.environ.get("TEKNOFEST_CONFIG", str(Path(_bera_root) / "config.yaml"))
        )

        observer = ObserverAgent(cfg.perception)
        engine = EventEngine(cfg.events, fps=fps)

        frames_rgb = []
        for fpath in frame_paths:
            if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
                continue
            img_bgr = cv2.imread(fpath)
            if img_bgr is None:
                continue
            frames_rgb.append(img_bgr[:, :, ::-1])  # BGR → RGB

        if not frames_rgb:
            return []

        observations = observer.observe_video(frames_rgb, fps=fps, sampled_indices=sampled_indices)
        for obs in observations:
            engine.process_observation(obs)

        return engine.get_signals()

    except Exception as exc:
        logger.error(f"Perception pipeline hatası: {exc}", exc_info=True)
        return []


async def handle_chunk(chunk_data: dict, redis_client):
    """Tek bir FrameChunk'ı işler; buffer dolunca algı pipeline'ını çalıştırır."""
    chunk = FrameChunk(**chunk_data)
    buf = _job_buffers[chunk.job_id]
    buf["paths"].extend(chunk.frame_paths)
    buf["indices"].extend(chunk.frame_indices)
    buf["fps"] = chunk.fps

    _metrics["processed_frames"] += len(chunk.frame_paths)

    # is_last=True → tüm bufferi işle ve job'ı temizle
    if chunk.is_last:
        logger.info(
            f"Processing job {chunk.job_id} — {len(buf['paths'])} frames"
        )
        loop = asyncio.get_event_loop()
        signals = await loop.run_in_executor(
            None,
            _run_perception_sync,
            buf["paths"],
            buf["indices"],
            buf["fps"],
        )

        for sig in signals:
            event = EventDetected(
                job_id=chunk.job_id,
                camera_id=chunk.camera_id,
                event_type=sig.get("event_type", "unknown"),
                timestamp=sig.get("timestamp", "00:00"),
                confidence=float(sig.get("confidence", 0.5)),
                description=sig.get("description", ""),
                # Kanal B'nin videoyu bütün olarak analiz edebilmesi için
                # kaynak yolu zincirin devamına taşı.
                video_path=chunk.video_path,
            )
            await redis_helper.publish_message(redis_client, "event.detected", event)
            _metrics["events_detected"] += 1

        _metrics["jobs_processed"] += 1
        del _job_buffers[chunk.job_id]
        logger.info(
            f"Job {chunk.job_id} done — {len(signals)} events published"
        )


async def redis_consumer():
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    client = await redis_helper.get_redis_client(redis_url)

    stream_name = "frame.chunk"
    group_name = "perception_group"
    consumer_name = "perception_consumer_1"

    try:
        await client.xgroup_create(stream_name, group_name, id='0', mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.warning(f"Group init error for {stream_name}: {e}")

    logger.info("Started Perception Service consumer...")
    while True:
        try:
            messages = await client.xreadgroup(
                groupname=group_name,
                consumername=consumer_name,
                streams={stream_name: ">"},
                count=5,
                block=2000,
            )
            for _, msgs in messages:
                for msg_id, msg_data in msgs:
                    raw = msg_data.get("payload", "{}")
                    try:
                        payload = json.loads(raw)
                        await handle_chunk(payload, client)
                        await redis_helper.ack_message(client, stream_name, group_name, msg_id)
                    except Exception as e:
                        logger.error(f"Chunk processing error: {e}", exc_info=True)
                        _metrics["dlq_count"] += 1
                        await redis_helper.publish_raw(client, f"{stream_name}.dlq", {
                            "msg_id": msg_id, "raw": raw, "error": str(e)
                        })
                        await redis_helper.ack_message(client, stream_name, group_name, msg_id)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Perception consumer error: {e}")
            await asyncio.sleep(1)

    await client.aclose()


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(redis_consumer())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="Dalga AI Perception Service", lifespan=lifespan)
app.include_router(
    create_health_router("perception-service", get_details_fn=lambda: _metrics),
    prefix="/api/v1",
)


@app.get("/api/v1/metrics")
async def get_metrics():
    return _metrics
