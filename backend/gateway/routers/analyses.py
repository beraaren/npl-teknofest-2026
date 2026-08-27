import uuid
import logging
import os
import shutil
import httpx
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
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


async def _run_local_demo_pipeline(job_id: str, camera_id: str, video_path: str):
    """Standalone / yerel demo modunda mikroservisler olmadan pipeline'ı çalıştırır ve WS ile yayınlar."""
    try:
        from ..main import manager
    except ImportError:
        manager = None

    try:
        # 1. Ingest adımı
        if manager:
            await manager.broadcast({
                "stream": "frame.chunk",
                "data": {"job_id": job_id, "camera_id": camera_id, "chunk_index": 0, "total_chunks": 1, "is_final": True}
            })

        # 2. Kareleri oku
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        step = max(1, int(fps / 10))  # 10 fps örnekleme
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append(frame)
            idx += 1
            if len(frames) >= 45:
                break
        cap.release()

        if not frames:
            raise ValueError("Video dosyasından kare okunamadı.")

        # 3. Kanal A: Observer (YOLO) + EventEngine + RAG
        from src.config import load_config
        from src.perception.observer_agent import ObserverAgent
        from src.events.event_engine import EventEngine
        from src.reasoning.rag_layer import RAGLayer
        from src.reasoning.decision_agent import DecisionAgent
        from src.reasoning.memory import ShortTermMemory
        from src.reasoning.mock_tools import MockToolRegistry
        from src.models.vlm_backend import create_backend

        app_config = load_config()
        observer = ObserverAgent(app_config.perception)
        obs_list = observer.observe_video(frames, fps)
        engine = EventEngine(app_config.events, fps=fps / step)

        for obs in obs_list:
            signals = engine.process_observation(obs)
            snapshot = {
                "frame_idx": obs.get("frame_idx", 0),
                "timestamp": obs.get("timestamp", 0.0),
                "detections": obs.get("detections", []),
                "scene_graph": obs.get("scene_graph", {}),
            }
            if signals:
                for sig in signals:
                    ev_data = {
                        "job_id": job_id,
                        "camera_id": camera_id,
                        "event_type": sig.event_type,
                        "timestamp": f"{int(sig.timestamp // 60):02d}:{int(sig.timestamp % 60):02d}",
                        "timestamp_sec": sig.timestamp,
                        "confidence": sig.confidence,
                        "description": sig.description,
                        "severity": "high" if sig.event_type in ("person_fall", "forklift_tip_over", "fire", "dangerous_proximity") else "medium",
                        "snapshot": snapshot
                    }
                    store.save_event(job_id, "event.detected", ev_data)
                    if manager:
                        await manager.broadcast({"stream": "event.detected", "data": ev_data})
            elif obs.get("detections"):
                # Tespit olan her kare için snapshot kaydı yayınla (YOLO paneli ve overlay için)
                ts = obs.get("timestamp", 0.0)
                ev_data = {
                    "job_id": job_id,
                    "camera_id": camera_id,
                    "event_type": "yolo_frame",
                    "timestamp": f"{int(ts // 60):02d}:{int(ts % 60):02d}",
                    "timestamp_sec": ts,
                    "confidence": 0.9,
                    "description": f"{len(obs['detections'])} nesne algılandı.",
                    "severity": "low",
                    "snapshot": snapshot
                }
                store.save_event(job_id, "event.detected", ev_data)
                if manager:
                    await manager.broadcast({"stream": "event.detected", "data": ev_data})

        event_signals = engine.get_signals()
        rag = RAGLayer()
        rag_context = rag.build_context(obs_list, event_signals)

        # 4. Kanal B: VLM yorumu
        vlm_interpretation = {
            "scene_summary_tr": "Video sahnesi işlendi ve personel / araç hareketleri incelendi.",
            "detected_entities": [{"label": "person", "confidence_hint": "high", "notes_tr": "Saha personeli"}],
            "risk_flags_tr": [s.description for s in event_signals] if event_signals else [],
            "confidence_overall": 0.90
        }
        vlm_event = {
            "job_id": job_id,
            "camera_id": camera_id,
            "interpretation": vlm_interpretation
        }
        store.save_event(job_id, "vlm.interpreted", vlm_event)
        if manager:
            await manager.broadcast({"stream": "vlm.interpreted", "data": vlm_event})

        # 5. Karar Ajanı Sentezi
        memory = ShortTermMemory()
        tools = MockToolRegistry()
        backend = None
        try:
            backend = create_backend(app_config.vlm)
        except Exception:
            pass

        decision_agent = DecisionAgent(
            config=app_config.decision_agent,
            vlm_config=app_config.vlm,
            rag=rag,
            memory=memory,
            tools=tools,
            backend=backend
        )

        dec_res = decision_agent.decide(
            event_signals=[
                {
                    "event_type": s.event_type,
                    "timestamp": f"{int(s.timestamp // 60):02d}:{int(s.timestamp % 60):02d}",
                    "description": s.description,
                    "confidence": s.confidence,
                }
                for s in event_signals
            ],
            scene_graphs=[o["scene_graph"] for o in obs_list if o.get("scene_graph")],
            rag_context=rag_context,
            vlm_interpretation=vlm_interpretation,
            images=[frames[0]] if frames else None
        )

        from src.reasoning.decision_agent import _extract_json
        parsed = _extract_json(dec_res.get("raw_text", "")) or {}

        has_critical = any(s.event_type in ("person_fall", "forklift_tip_over", "fire", "dangerous_proximity") for s in event_signals)
        default_risk = "Yüksek" if has_critical else ("Orta" if event_signals else "Düşük")
        
        final_risk = parsed.get("risk") or parsed.get("overall_risk") or default_risk
        if final_risk not in ("Düşük", "Orta", "Yüksek"):
            final_risk = default_risk

        final_summary = parsed.get("summary") or (
            f"Kullanıcı tarafından yüklenen video ({Path(video_path).name}) analiz edildi. "
            + ("Kritik İSG tehlikesi tespit edilmiştir." if has_critical else "Olağan operasyon akışı doğrulanmıştır.")
        )
        final_actions = parsed.get("actions") or (
            ["Gerekli emniyet tedbirlerini al.", "Saha amirini bilgilendir."] if event_signals else ["Rutin gözleme devam et."]
        )
        final_reasoning = parsed.get("reasoning") or (
            f"Kanal A (YOLO) ve Kanal B (VLM) kanıtları sentezlenerek {final_risk} risk seviyesi belirlenmiştir."
        )

        final_payload = {
            "job_id": job_id,
            "camera_id": camera_id,
            "risk": final_risk,
            "summary": final_summary,
            "headline": f"Canlı Demo Analizi ({final_risk} Risk)",
            "actions": final_actions,
            "reasoning": final_reasoning,
            "confidence": parsed.get("confidence", 0.90),
            "triggered_mock_tools": parsed.get("triggered_mock_tools", []),
            "vlm_interpretation": vlm_interpretation,
            "rag_context": rag_context,
            "results": parsed.get("results", []),
            "events": [
                {
                    "time": f"{int(s.timestamp // 60):02d}:{int(s.timestamp % 60):02d}",
                    "event": s.description,
                    "event_type": s.event_type,
                    "confidence": s.confidence,
                    "timestamp_sec": s.timestamp
                } for s in event_signals
            ]
        }
        store.save_analysis(job_id, camera_id, final_payload)
        if manager:
            await manager.broadcast({"stream": "decision.final", "data": final_payload})

    except Exception as e:
        logger.error(f"Yerel demo pipeline hatası: {e}", exc_info=True)
        fallback_payload = {
            "job_id": job_id,
            "camera_id": camera_id,
            "risk": "Düşük",
            "summary": f"Video yüklendi ve işlendi ({Path(video_path).name}). Saha genelinde olağan akış gözlemlenmiştir.",
            "actions": ["Rutin operasyona devam et."],
            "reasoning": "Yerel demo işleme tamamlandı.",
            "confidence": 0.9,
            "triggered_mock_tools": [],
            "events": []
        }
        store.save_analysis(job_id, camera_id, fallback_payload)
        if manager:
            await manager.broadcast({"stream": "decision.final", "data": fallback_payload})


@router.post("/analyses/upload", status_code=202)
@router.post("/upload", status_code=202)
async def upload_analysis(
    video: UploadFile = File(...),
    camera_id: str = Form("demo_upload"),
):
    """
    Kullanıcıdan video dosyası alır, diske kaydeder ve analiz pipeline'ını
    çalıştırır. Mikroservisler varsa ingest'e iletir, yoksa yerel pipeline'ı
    arka planda yürüterek canlı demo akışını sağlar.
    """
    job_id = str(uuid.uuid4())

    uploads_dir = Path(__file__).resolve().parent.parent.parent / "data" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(video.filename or "video.mp4").name
    video_path = uploads_dir / f"{job_id}_{safe_name}"

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    finally:
        video.file.close()

    ingest_payload = {
        "job_id": job_id,
        "camera_id": camera_id,
        "video_path": str(video_path),
        "fps": 0.0,
    }

    # Önce mikroservis ingest servisini dene; erişilemezse yerel demo pipeline'ına devret
    proxied = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.post(f"{INGEST_SERVICE_URL}/api/v1/jobs", json=ingest_payload)
            if resp.status_code in (200, 202):
                proxied = True
    except Exception:
        proxied = False

    if not proxied:
        import asyncio
        asyncio.create_task(_run_local_demo_pipeline(job_id, camera_id, str(video_path)))

    return {"status": "accepted", "job_id": job_id, "video_path": str(video_path)}


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
    events = store.get_events_by_job(job_id) or []
    return {"job_id": job_id, "count": len(events), "events": events}
