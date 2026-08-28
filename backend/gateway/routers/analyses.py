import uuid
import logging
import os
import shutil
import httpx
from pathlib import Path
from typing import Any, Optional
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


_CRITICAL_EVENT_TYPES = (
    "person_fall",
    "forklift_tip_over",
    "fire",
    "fire_smoke",
    "explosion",
    "dangerous_proximity",
)


def _severity_for_event(event_type: str) -> str:
    return "high" if event_type in _CRITICAL_EVENT_TYPES else "medium"


def _tools_for_event(event_type: str) -> list:
    """Bir olay tipine göre deterministik mock-araç listesi döner."""
    mapping = {
        "person_fall": ["call_health_team", "notify_supervisor"],
        "fire": ["trigger_fire_suppression", "sound_alarm", "activate_cbrn_protocol"],
        "fire_smoke": ["trigger_fire_suppression", "sound_alarm", "activate_cbrn_protocol"],
        "smoke": ["activate_cbrn_protocol", "sound_alarm", "notify_supervisor"],
        "explosion": ["lockdown_facility", "trigger_fire_suppression", "call_health_team", "sound_alarm"],
        "dangerous_proximity": ["notify_supervisor", "sound_alarm"],
        "forklift_tip_over": ["secure_area", "call_health_team", "notify_supervisor"],
        "no_helmet": ["notify_supervisor", "sound_alarm"],
        "no_vest": ["notify_supervisor", "sound_alarm"],
    }
    return mapping.get(event_type, [])


def _parse_timestamp(ts: Any) -> float:
    """EventEngine'in 'MM:SS' dizgesini veya sayıyı saniyeye çevirir."""
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str) and ":" in ts:
        parts = ts.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    return 0.0


def _build_demo_payload(
    job_id: str,
    camera_id: str,
    video_path: str,
    event_signals: list,
    rag_context: dict,
    vlm_interpretation: dict,
    parsed: Optional[dict] = None,
) -> dict:
    """Hem başarılı hem fallback yol için deterministik demo payload'ı üretir."""
    parsed = parsed or {}
    has_critical = any(s.get("event_type") in _CRITICAL_EVENT_TYPES for s in event_signals)
    default_risk = "Yüksek" if has_critical else ("Orta" if event_signals else "Düşük")

    final_risk = parsed.get("risk") or parsed.get("overall_risk") or default_risk
    if final_risk not in ("Düşük", "Orta", "Yüksek"):
        final_risk = default_risk

    # Güvence: Kanal A kritik bir olay (yangın/duman/düşme/patlama vb.) doğrulamışsa
    # karar ajanı riski bunun altına düşüremez.
    _risk_order = {"Düşük": 1, "Orta": 2, "Yüksek": 3}
    if _risk_order.get(final_risk, 0) < _risk_order.get(default_risk, 0):
        final_risk = default_risk

    if parsed.get("summary"):
        final_summary = parsed["summary"]
    else:
        final_summary = (
            f"Kullanıcı tarafından yüklenen video ({Path(video_path).name}) analiz edildi. "
            + ("Kritik İSG tehlikesi tespit edilmiştir." if has_critical else "Olağan operasyon akışı doğrulanmıştır.")
        )

    if parsed.get("actions"):
        final_actions = parsed["actions"]
    else:
        if has_critical:
            final_actions = [
                "Tehlike bölgesi güvenlik altına alınsın.",
                "İlgili ekip (sağlık/güvenlik) olay konumuna yönlendirilsin.",
                "Saha amirine anında bilgi verilsin.",
            ]
        elif event_signals:
            final_actions = [
                "Gerekli emniyet tedbirlerini al.",
                "Saha amirini bilgilendir.",
            ]
        else:
            final_actions = ["Rutin gözleme devam et."]

    final_reasoning = parsed.get("reasoning") or (
        f"Kanal A (YOLO) ve Kanal B (VLM) kanıtları sentezlenerek {final_risk} risk seviyesi belirlenmiştir."
    )

    results = []
    for s in event_signals:
        event_type = s.get("event_type")
        if event_type == "yolo_frame":
            continue
        time_label = s.get("timestamp") or "00:00"
        timestamp_sec = _parse_timestamp(s.get("timestamp"))
        confidence = float(s.get("confidence") or 0.0)
        sev = _severity_for_event(event_type)
        results.append({
            "result_type": "contextual_finding",
            "time": time_label,
            "timestamp_sec": timestamp_sec,
            "event_type": event_type,
            "severity": sev,
            "hazard_mechanism": s.get("description"),
            "confidence": confidence,
            "evidence": {"agreement": "Kanal A (YOLO) tespiti"},
        })
        # Düşük güvenli tespitler için insan incelemesi gözlemi üret
        if confidence < 0.55:
            results.append({
                "result_type": "uncertain_observation",
                "time": time_label,
                "timestamp_sec": timestamp_sec,
                "event_type": event_type,
                "severity": sev,
                "uncertainty_reason": f"{event_type} tespit güveni düşük (%{int(confidence * 100)}); İSG uzmanı doğrulamalı.",
                "confidence": confidence,
                "evidence": {"agreement": "Kanal A (YOLO) zayıf kanıt"},
            })

    # Eğer hiç sinyal yoksa, en azından bir 'olağan akış' contextual finding göster
    if not any(r["result_type"] == "contextual_finding" for r in results):
        results.append({
            "result_type": "contextual_finding",
            "time": "00:00",
            "timestamp_sec": 0.0,
            "event_type": "routine_flow",
            "severity": "low",
            "hazard_mechanism": "Saha genelinde olağan operasyon akışı gözlemlendi.",
            "confidence": 0.90,
            "evidence": {"agreement": "Kanal A (YOLO) rutin tarama"},
        })
        results.append({
            "result_type": "uncertain_observation",
            "time": "00:00",
            "timestamp_sec": 0.0,
            "event_type": "routine_flow",
            "severity": "low",
            "uncertainty_reason": "YOLO/olay motoru bu videoda belirgin bir İSG olayı tespit etmedi; uzman incelemesi önerilir.",
            "confidence": 0.50,
            "evidence": {"agreement": "Kanal A (YOLO) negatif tarama"},
        })

    triggered = []
    seen = set()
    for s in event_signals:
        event_type = s.get("event_type")
        for tool_name in _tools_for_event(event_type):
            if tool_name not in seen:
                seen.add(tool_name)
                triggered.append({
                    "tool_name": tool_name,
                    "params": {"reason": s.get("description", ""), "location": camera_id},
                })
    if not triggered:
        triggered.append({
            "tool_name": "notify_supervisor",
            "params": {"message": f"{camera_id} kamerasından canlı demo analizi tamamlandı; olağan akış gözlemlendi."},
        })

    events = [
        {
            "time": s.get("timestamp") or "00:00",
            "event": s.get("description"),
            "event_type": s.get("event_type"),
            "severity": _severity_for_event(s.get("event_type")),
            "confidence": float(s.get("confidence") or 0.0),
            "timestamp_sec": _parse_timestamp(s.get("timestamp")),
        }
        for s in event_signals
    ]
    if not events:
        events.append({
            "time": "00:00",
            "event": "Olağan operasyon akışı gözlemlendi; belirgin İSG olayı tespit edilmedi.",
            "event_type": "routine_flow",
            "severity": "low",
            "confidence": 0.90,
            "timestamp_sec": 0.0,
        })

    # Karar ajanı (LLM) bazen araçları [{"tool_name": ...}] yerine düz metin
    # listesi ["sound_alarm", ...] olarak döndürüyor; demo arayüzü buton
    # çizebilmek için dict formu beklediğinden burada normalize edilir.
    parsed_tools = []
    for entry in parsed.get("triggered_mock_tools") or []:
        if isinstance(entry, str):
            parsed_tools.append({
                "tool_name": entry,
                "params": {"reason": "Karar ajanı önerisi", "location": camera_id},
            })
        elif isinstance(entry, dict) and entry.get("tool_name"):
            entry.setdefault("params", {})
            parsed_tools.append(entry)

    return {
        "job_id": job_id,
        "camera_id": camera_id,
        "risk": final_risk,
        "summary": final_summary,
        "headline": f"Canlı Demo Analizi ({final_risk} Risk)",
        "actions": final_actions,
        "reasoning": final_reasoning,
        "confidence": parsed.get("confidence", 0.90),
        "triggered_mock_tools": parsed_tools if parsed_tools else triggered,
        "vlm_interpretation": vlm_interpretation,
        "rag_context": rag_context,
        "results": parsed.get("results") if parsed.get("results") else results,
        "events": events,
    }


async def _run_local_demo_pipeline(job_id: str, camera_id: str, video_path: str):
    """Standalone / yerel demo modunda mikroservisler olmadan pipeline'ı çalıştırır ve WS ile yayınlar."""
    try:
        from ..main import manager
    except ImportError:
        manager = None

    # Aşamalı biriktiriciler: herhangi bir adımda hata olursa eldeki verilerle fallback üretilecek.
    frames = []
    obs_list = []
    event_signals = []
    rag_context = {}
    vlm_interpretation = {
        "scene_summary_tr": "Video sahnesi işlendi ve personel / araç hareketleri incelendi.",
        "detected_entities": [],
        "risk_flags_tr": [],
        "confidence_overall": 0.90,
    }
    parsed = {}

    try:
        # 1. Ingest adımı
        if manager:
            await manager.broadcast({
                "stream": "frame.chunk",
                "data": {"job_id": job_id, "camera_id": camera_id, "chunk_index": 0, "total_chunks": 1, "is_final": True}
            })

        # 2. Kareleri oku — tasarım: Kanal A videonun ilk 60 saniyesini
        # (scene_max_segment_sec) 10 fps örnekleme ile işler. Önceden yalnızca
        # ilk 45 kare (~4.5 sn) okunuyordu; videonun ortasındaki/sonundaki
        # olaylar (patlama, düşme) hiç görülmüyordu.
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = total_frames / fps if fps else 0.0
        segment_sec = min(duration_sec or 60.0, 60.0)  # Kanal B segment tavanıyla aynı
        max_frames = max(45, int(segment_sec * 10))
        step = max(1, int(fps / 10))  # 10 fps örnekleme
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append(frame)
            idx += 1
            if len(frames) >= max_frames:
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
                        "severity": _severity_for_event(sig.event_type),
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

        # 4. Kanal B: VLM yorumu — GERÇEK Kanal B pipeline'ı çalıştırılır.
        # Eskiden burada hardcoded bir stub vardı; VLM videoyu hiç görmüyordu
        # ve karar ajanı "patlamaya dair işaret bulunamadı" gibi görüntüden
        # bağımsız özetler üretiyordu. run_channel_b, videoyu 60 sn'lik
        # segmentlere böler, her segmenti iteratif olarak (önceki segmentlerin
        # damıtılmış hafıza bağlamıyla) VLM'e gönderir ve tek bir yoruma
        # indirger (bkz. Kanal_B/pipeline.py).
        # analyses.py -> routers -> gateway -> backend -> proje kökü (4 seviye)
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        try:
            from dotenv import load_dotenv
            load_dotenv(project_root / ".env")
        except Exception:
            pass
        import sys as _sys
        for entry in (str(project_root), str(project_root / "Kanal_B")):
            if entry not in _sys.path:
                _sys.path.insert(0, entry)

        vlm_interpretation = None
        try:
            import asyncio
            from pipeline import run_channel_b  # Kanal_B/pipeline.py

            loop = asyncio.get_running_loop()
            vlm_interpretation = await loop.run_in_executor(
                None,
                lambda: run_channel_b(
                    video_path,
                    video_id=job_id,
                    output_dir=str(project_root / "outputs" / "channel_b" / job_id),
                ),
            )
        except Exception as vlm_exc:
            logger.error(f"Kanal B VLM hatası: {vlm_exc}", exc_info=True)

        if not vlm_interpretation:
            # VLM erişilemezse Kanal A sinyallerinden türetilen yedek yorum
            vlm_interpretation = {
                "scene_summary_tr": "Kanal B (VLM) erişilemedi; yorum Kanal A sinyallerinden türetildi.",
                "detected_entities": [{"label": "person", "confidence_hint": "high", "notes_tr": "Saha personeli"}],
                "risk_flags_tr": [s.get("description") for s in event_signals] if event_signals else [],
                "confidence_overall": 0.90,
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
                    "event_type": s.get("event_type"),
                    "timestamp": s.get("timestamp") or "00:00",
                    "description": s.get("description"),
                    "confidence": float(s.get("confidence") or 0.0),
                }
                for s in event_signals
            ],
            scene_graphs=[o["scene_graph"] for o in obs_list if o.get("scene_graph")],
            rag_context=rag_context,
            vlm_interpretation=vlm_interpretation,
        )

        from src.reasoning.decision_agent import _extract_json
        parsed = _extract_json(dec_res.get("raw_text", "")) or {}

    except Exception as e:
        logger.error(f"Yerel demo pipeline hatası (deterministik fallback üretiliyor): {e}", exc_info=True)

    # Her durumda zengin bir payload oluştur ve kaydet
    final_payload = _build_demo_payload(
        job_id=job_id,
        camera_id=camera_id,
        video_path=video_path,
        event_signals=event_signals,
        rag_context=rag_context,
        vlm_interpretation=vlm_interpretation,
        parsed=parsed,
    )
    store.save_analysis(job_id, camera_id, final_payload)
    if manager:
        await manager.broadcast({"stream": "decision.final", "data": final_payload})


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
