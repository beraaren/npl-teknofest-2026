"""vlm-service — event.detected tüketip vlm.interpreted yayınlar.

§2.3.3 tasarımına uygun:
  - event.detected sinyali gelince tetiklenir
  - select_critical_frames ile kritik kareler seçilir
  - run_channel_b çağrılır; başarısız olursa interpret_frames fallback
  - Kanal B bağımsızlığı: RAG/sahne grafi VERİLMEZ (birleştirme decision'da)
  - GPU kullanan tek servis — run_in_executor ile non-blocking
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
_metrics = {"vlm_calls": 0, "vlm_errors": 0, "fallback_used": 0, "dlq_count": 0}

# Job başına frame bilgisi: {job_id: {"paths": [], "indices": [], "fps": float}}
# perception-service ile aynı frame yollarına erişilir (paylaşılan disk)
_job_frames: dict = {}


def _load_frames_from_paths(frame_paths: list[str]) -> list:
    """Frame yollarından RGB numpy dizilerini yükler."""
    import cv2
    frames = []
    for fpath in frame_paths:
        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
            continue
        img = cv2.imread(fpath)
        if img is not None:
            frames.append(img[:, :, ::-1])  # BGR → RGB
    return frames


def _run_vlm_sync(frame_paths: list[str], sampled_indices: list[int],
                  event_signals: list[dict], fps: float, job_id: str = "") -> dict:
    """
    Senkron GPU işlevi — run_in_executor içinde çalıştırılır.
    1. select_critical_frames ile kritik kareler seçilir.
    2. Kanal B pipeline / backend infer öncelikli olarak çalıştırılır.
    3. Başarısız olursa DecisionAgent.interpret_frames fallback olarak devreye girer.
    """
    import sys
    _bera_root = str(Path(__file__).resolve().parent.parent.parent)
    if _bera_root not in sys.path:
        sys.path.insert(0, _bera_root)

    try:
        import cv2
        from src.preprocessing.critical_frames import select_critical_frames
        cfg_path = os.environ.get("TEKNOFEST_CONFIG", str(Path(_bera_root) / "config.yaml"))

        frames_rgb = _load_frames_from_paths(frame_paths)
        if not frames_rgb:
            return {
                "scene_summary_tr": "Kare yüklenemedi.",
                "confidence_overall": 0.0,
                "risk_flags_tr": [],
                "notable_frames": [],
            }

        max_count = 4  # config'den okunabilir
        critical_frames, critical_indices = select_critical_frames(
            frames_rgb, sampled_indices, event_signals, fps, max_count=max_count
        )
        if not critical_frames:
            critical_frames = frames_rgb[:max_count]
            critical_indices = sampled_indices[:max_count]

        # 1. ÖNCELİK: Kanal B Backend + Grid Paketi ile infer
        try:
            sys.path.insert(0, str(Path(_bera_root) / "Kanal_B"))
            from Kanal_B.backend import build_backend, _load_vlm_config
            from Kanal_B.contracts import (
                VLMFramePacket, FrameMeta, FrameQualityMetrics,
                GridLayout, EnhancementInfo
            )

            vlm_cfg = _load_vlm_config()
            backend = build_backend(vlm_cfg)

            # Grid görüntüsünü oluştur ve diske yaz
            grid_cols = min(4, len(critical_frames))
            grid_rows = (len(critical_frames) + grid_cols - 1) // grid_cols
            cell_w, cell_h = 384, 384
            cells = [cv2.resize(f, (cell_w, cell_h), interpolation=cv2.INTER_AREA) for f in critical_frames]
            while len(cells) < grid_rows * grid_cols:
                cells.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))
            
            row_blocks = [np.hstack(cells[r * grid_cols:(r + 1) * grid_cols]) for r in range(grid_rows)]
            grid_img = np.vstack(row_blocks)

            out_dir = os.path.join("data", "frames", job_id or "vlm_out")
            os.makedirs(out_dir, exist_ok=True)
            grid_path = os.path.join(out_dir, f"vlm_grid_{job_id}.jpg")
            cv2.imwrite(grid_path, cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])

            frame_metas = [
                FrameMeta(
                    frame_index=idx,
                    timestamp_sec=float(idx) / max(fps, 1.0),
                    grid_position=i,
                    selection_reason="scene_change",
                    quality=FrameQualityMetrics(laplacian_var=100.0, ssim_diff=0.5, brightness_mean=120.0),
                )
                for i, idx in enumerate(critical_indices)
            ]

            packet = VLMFramePacket(
                video_id=job_id,
                source_start_sec=frame_metas[0].timestamp_sec if frame_metas else 0.0,
                source_end_sec=frame_metas[-1].timestamp_sec if frame_metas else 0.0,
                frames=frame_metas,
                grid_layout=GridLayout(rows=grid_rows, cols=grid_cols, cell_size=(cell_w, cell_h)),
                enhancement=EnhancementInfo(clahe_applied=False, clip_limit=2.5, tile_grid_size=(8, 8)),
                grid_image_path=grid_path,
            )

            interpretation = backend.infer(packet)
            res = interpretation.to_dict()
            res["notable_frames"] = critical_indices

            # risk_flags_tr uyumluluğunu sağla
            if not res.get("risk_flags_tr") and res.get("risk_events"):
                res["risk_flags_tr"] = [
                    r.get("description_tr", "")
                    for r in res["risk_events"]
                    if isinstance(r, dict) and r.get("description_tr")
                ]

            logger.info("VLM interpretation generated via Kanal_B backend successfully.")
            return res

        except Exception as kb_err:
            logger.warning(f"Kanal_B infer başarısız ({kb_err}), DecisionAgent.interpret_frames fallback'e geçiliyor")
            _metrics["fallback_used"] += 1
            from src.config import load_config
            from src.reasoning.decision_agent import DecisionAgent
            from src.reasoning.rag_layer import RAGLayer
            from src.reasoning.memory import ShortTermMemory
            from src.reasoning.mock_tools import MockToolRegistry

            cfg = load_config(cfg_path)
            rag = RAGLayer()
            memory = ShortTermMemory()
            tools = MockToolRegistry()
            agent = DecisionAgent(cfg.decision_agent, cfg.vlm, rag, memory, tools)
            out = agent.interpret_frames(critical_frames)
            out["notable_frames"] = critical_indices
            return out

    except Exception as exc:
        logger.error(f"VLM pipeline hatası: {exc}", exc_info=True)
        return {
            "scene_summary_tr": f"VLM yorumu başarısız: {exc}",
            "confidence_overall": 0.0,
            "risk_flags_tr": [],
            "notable_frames": [],
        }


async def process_event(event_data: dict, redis_client):
    """Tek bir EventDetected mesajını işler → VLM yorumu üretir."""
    event = EventDetected(**event_data)
    logger.info(f"VLM interpreting {event.event_type} for job {event.job_id}")

    # Job'a ait frame yollarını bul (ingest servisinin yazdığı dizin)
    frames_dir = os.path.join("data", "frames", event.job_id)
    if os.path.exists(frames_dir):
        frame_paths = sorted([
            str(Path(frames_dir) / f)
            for f in os.listdir(frames_dir)
            if f.endswith(".jpg") and not f.startswith("vlm_grid")
        ])
    else:
        frame_paths = []

    if not frame_paths:
        logger.warning(f"Frames not found for job {event.job_id}, emitting empty VLM result")
        result = {"scene_summary_tr": "Kare bulunamadı.", "confidence_overall": 0.0, "risk_flags_tr": [], "notable_frames": []}
        critical_indices = []
    else:
        sampled_indices = list(range(len(frame_paths)))
        event_signals = [event_data]

        cfg = load_app_config()
        fps = cfg.preprocessing.channel_a_fps if cfg else 12.0

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            _run_vlm_sync,
            frame_paths,
            sampled_indices,
            event_signals,
            fps,
            event.job_id,
        )
        critical_indices = result.get("notable_frames", [])
        _metrics["vlm_calls"] += 1

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

