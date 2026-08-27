"""vlm-service — event.detected tüketip vlm.interpreted yayınlar.

§2.3.3 tasarımına uygun:
  - event.detected sinyali gelince tetiklenir
  - Kanal B, videoyu EVREN'in video modeline (alias "vlm") BÜTÜN olarak gönderir;
    60 saniyeyi aşan videolar 720p/60sn segmentlere bölünüp sırayla incelenir ve
    segmentler arası bağlam metin tabanlı hafızayla taşınır (bkz. Kanal_B/pipeline.py)
  - Video yolu yoksa veya video modu başarısız olursa HATA döner; mock/kare fallback yoktur.
  - Kanal B bağımsızlığı: RAG/sahne grafi VERİLMEZ (birleştirme decision'da)
  - Ağ/GPU bekleyen tek servis — run_in_executor ile non-blocking
"""
import asyncio
import json
import os
import logging
from pathlib import Path
import numpy as np
from fastapi import FastAPI
from contextlib import asynccontextmanager

from ..common.health import create_health_router
from ..common import redis as redis_helper
from ..common.config_loader import load_app_config
from ..contracts.messages import EventDetected, VlmInterpreted

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrik sayaçları
_metrics = {
    "vlm_calls": 0,
    "vlm_errors": 0,
    "video_mode": 0,
    "dlq_count": 0,
}

# Aynı job için Kanal B'yi bir kez çalıştır: event.detected job başına birden
# fazla kez gelir (her olay sinyali için), ama Kanal B tüm videoyu analiz eder.
# Bu küme olmadan aynı video her sinyalde yeniden analiz edilir ve hem süre hem
# sistem yükü gereksiz katlanır.
_analyzed_jobs: set[str] = set()


def _ensure_kanal_b_on_path() -> str:
    """Proje kökünü ve ``Kanal_B/`` dizinini ``sys.path``e ekler.

    ``Kanal_B`` modülleri birbirini paket öneki olmadan içe aktarır
    (``from contracts import ...``); bu, ``Kanal_B/`` dizininin doğrudan
    ``sys.path``te olmasını gerektirir. test_akis.py de aynı deseni kullanır.
    """
    import sys

    root = str(Path(__file__).resolve().parent.parent.parent)
    kanal_b = str(Path(root) / "Kanal_B")
    for entry in (root, kanal_b):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return root


def _run_video_mode_sync(video_path: str, job_id: str) -> dict:
    """Kanal B'yi video modunda çalıştırır (birincil yol).

    Uzun videolar :func:`Kanal_B.pipeline.run_channel_b` içinde otomatik olarak
    segmentlenir; bu servis segment yönetimiyle ilgilenmez.

    Args:
        video_path: Kaynak video dosyası.
        job_id: İzlenebilirlik kimliği; çıktı klasörü adı olarak da kullanılır.

    Returns:
        S8 sözleşmesine uygun yorum sözlüğü.
    """
    _ensure_kanal_b_on_path()
    from pipeline import run_channel_b  # Kanal_B/pipeline.py

    out_dir = os.path.join("data", "channel_b", job_id)
    return run_channel_b(video_path, video_id=job_id, output_dir=out_dir)


def _run_vlm_sync(
    video_path: str,
    frame_paths: list[str],
    sampled_indices: list[int],
    event_signals: list[dict],
    fps: float,
    job_id: str,
) -> dict:
    """Kanal B'yi çalıştırır: yalnızca video modu.

    Video yolu yoksa veya video modu başarısız olursa exception yükseltir;
    mock/kare fallback kullanılmaz. Çağıran bu hatayı dlq'ya veya log'a yazar.
    """
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"Video bulunamadı: {video_path}")

    result = _run_video_mode_sync(video_path, job_id)
    _metrics["video_mode"] += 1
    logger.info(
        f"Kanal B video modunda tamamlandı (job={job_id}, "
        f"segment={result.get('segment_count', 1)})"
    )
    return result


async def process_event(event_data: dict, redis_client):
    """Tek bir EventDetected mesajını işler → VLM yorumu üretir."""
    event = EventDetected(**event_data)

    # Aynı job için Kanal B'yi tekrar çalıştırma (video analizi olay başına değil,
    # video başına yapılır).
    if event.job_id in _analyzed_jobs:
        logger.info(f"Job {event.job_id} için Kanal B zaten çalıştı, atlanıyor.")
        return
    _analyzed_jobs.add(event.job_id)

    logger.info(f"VLM interpreting {event.event_type} for job {event.job_id}")

    frames_dir = os.path.join("data", "frames", event.job_id)
    if os.path.exists(frames_dir):
        frame_paths = sorted(
            str(Path(frames_dir) / f)
            for f in os.listdir(frames_dir)
            if f.endswith(".jpg") and not f.startswith("vlm_grid")
        )
    else:
        frame_paths = []

    if not event.video_path and not frame_paths:
        raise FileNotFoundError(f"Job {event.job_id}: ne video ne kare bulundu")

    cfg = load_app_config()
    fps = cfg.preprocessing.channel_a_fps if cfg else 12.0

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        _run_vlm_sync,
        event.video_path,
        frame_paths,
        list(range(len(frame_paths))),
        [event_data],
        fps,
        event.job_id,
    )
    critical_indices = result.get("notable_frames", []) or []
    _metrics["vlm_calls"] += 1

    # risk_flags_tr uyumluluğu: yeni üreticiler risk_events doldurur, eski
    # tüketiciler düz metin listesi okur.
    if not result.get("risk_flags_tr") and result.get("risk_events"):
        result["risk_flags_tr"] = [
            r.get("description_tr", "")
            for r in result["risk_events"]
            if isinstance(r, dict) and r.get("description_tr")
        ]

    vlm_msg = VlmInterpreted(
        job_id=event.job_id,
        camera_id=event.camera_id,
        interpretation=result,
        critical_indices=critical_indices if isinstance(critical_indices, list) else [],
    )
    await redis_helper.publish_message(redis_client, "vlm.interpreted", vlm_msg)


async def redis_consumer():
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    client = await redis_helper.get_redis_client(redis_url)

    stream_name = "event.detected"
    group_name = "vlm_group"
    consumer_name = "vlm_consumer_1"

    try:
        await client.xgroup_create(stream_name, group_name, id='0', mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.warning(f"Group init error for {stream_name}: {e}")

    logger.info("Started VLM Service consumer...")
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
                        await process_event(payload, client)
                        # Başarılı işlemde ACK ver
                        await redis_helper.ack_message(client, stream_name, group_name, msg_id)
                    except json.JSONDecodeError as decode_err:
                        logger.error(f"VLM JSON decode error for {msg_id}: {decode_err}")
                        _metrics["dlq_count"] += 1
                        await redis_helper.publish_raw(client, f"{stream_name}.dlq", {
                            "msg_id": msg_id, "raw": raw, "error": str(decode_err)
                        })
                        await redis_helper.ack_message(client, stream_name, group_name, msg_id)
                    except Exception as e:
                        # §2.9: VLM / sistem çökmesi durumunda mesaj ACK'lenmez, pending'de kalır
                        logger.error(f"VLM processing error (message kept in pending): {e}", exc_info=True)
                        _metrics["vlm_errors"] += 1
                        _metrics["dlq_count"] += 1
                        await redis_helper.publish_raw(client, f"{stream_name}.dlq", {
                            "msg_id": msg_id, "error": str(e)
                        })
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"VLM consumer loop error: {e}")
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

app = FastAPI(title="Dalga AI VLM Service", lifespan=lifespan)
app.include_router(
    create_health_router("vlm-service", get_details_fn=lambda: _metrics),
    prefix="/api/v1",
)


@app.get("/api/v1/metrics")
async def get_metrics():
    return _metrics
