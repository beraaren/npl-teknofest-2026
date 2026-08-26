"""decision-service — event.detected + vlm.interpreted tüketip decision.final yayınlar.

§2.3.4 tasarımına uygun:
  - event.detected + vlm.interpreted mesajlarını job bazında birleştirir
  - RAGLayer risk kataloğunu sorgular
  - ShortTermMemory geçmiş olayları hatırlar
  - DecisionAgent kanıtları VLM'e gönderir
  - OutputGuardrail şema doğrulama + kademeli retry (0.15→0.10→0.05)
  - VLM yorumu olmadan da karar üretilebilir (tek kaynak notu)
  - Pipeline hiçbir zaman sessizce ölmez; hata durumunda null_response yayınlanır
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
from ..contracts.messages import EventDetected, VlmInterpreted, DecisionFinal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrik sayaçları
_metrics = {"decisions_made": 0, "guardrail_retries": 0, "null_responses": 0, "dlq_count": 0}

# Job başına bağlam: {job_id: {"events": [], "vlm": [], "camera_id": str, "lock": asyncio.Lock(), "decided": bool}}
_job_context: dict = defaultdict(lambda: {
    "events": [],
    "vlm": [],
    "camera_id": "",
    "vlm_ready": asyncio.Event(),
    "lock": asyncio.Lock(),
    "decided": False,
})


def _run_decision_sync(events: list[dict], vlm_interpretations: list[dict]) -> dict:
    """
    Senkron karar işlevi — run_in_executor içinde çalıştırılır.
    RAGLayer + DecisionAgent + OutputGuardrail zincirini çalıştırır.
    """
    import sys
    _bera_root = str(Path(__file__).resolve().parent.parent.parent)
    if _bera_root not in sys.path:
        sys.path.insert(0, _bera_root)

    try:
        from src.config import load_config
        from src.reasoning.decision_agent import DecisionAgent
        from src.reasoning.rag_layer import RAGLayer
        from src.reasoning.memory import ShortTermMemory
        from src.reasoning.mock_tools import MockToolRegistry
        from src.output.guardrail import OutputGuardrail

        cfg_path = os.environ.get("TEKNOFEST_CONFIG", str(Path(_bera_root) / "config.yaml"))
        cfg = load_config(cfg_path)

        # Bileşenler: tasarım §1.1 — mevcut src modülleri sarılır
        rag = RAGLayer()
        memory = ShortTermMemory()
        tools = MockToolRegistry()
        guardrail = OutputGuardrail(cfg.output.guardrail)
        agent = DecisionAgent(cfg.decision_agent, cfg.vlm, rag, memory, tools)

        # RAG bağlamı oluştur
        vlm_flags = []
        if vlm_interpretations:
            vlm_flags = vlm_interpretations[0].get("risk_flags_tr", [])

        rag_context = rag.build_context(
            observation_report=[],
            event_signals=events,
            vlm_flags=vlm_flags,
        )

        vlm_interp = vlm_interpretations[0] if vlm_interpretations else None
        single_source = vlm_interp is None

        # Hafızaya olayları ekle
        for ev in events:
            memory.add(ev, entry_type="event")

        # Karar üret
        decision_raw = agent.decide(
            event_signals=events,
            scene_graphs=[],
            rag_context=rag_context,
            vlm_interpretation=vlm_interp,
        )

        raw_text = decision_raw.get("raw_text", "") if isinstance(decision_raw, dict) else str(decision_raw)
        retry_fn = decision_raw.get("retry_fn", lambda temp: raw_text) if isinstance(decision_raw, dict) else (lambda temp: raw_text)
        rag_risk_level = decision_raw.get("rag_risk_level", "Düşük") if isinstance(decision_raw, dict) else "Düşük"

        if single_source:
            # §2.9: VLM yokken "tek kaynak" notu
            raw_text = raw_text + (
                "\n[NOT: VLM yorumu alınamadı. Karar yalnızca geometrik sinyaller ve "
                "RAG risk kataloğuna dayanmaktadır.]"
            )

        # Guardrail doğrula + retry
        result = guardrail.validate(
            raw_text=raw_text,
            generate_fn=retry_fn,
            rag_risk_level=rag_risk_level,
        )
        # Guardrail başarısız olduğunda ÖZET BOŞ DÖNMEZ; yapılandırılmış
        # null_response metnini ("Bilmiyorum") döndürür. Bu yüzden çağıran
        # taraftaki 'if not summary' kontrolü null yanıtları hiç yakalayamıyor
        # ve başarılı karar olarak sayıyordu. Tespiti burada, yapılandırılmış
        # değerle karşılaştırarak yapıp sonuca işaretliyoruz.
        result["null_response"] = (
            str(result.get("summary", "")).strip()
            == str(cfg.output.guardrail.null_response).strip()
        )
        return result

    except Exception as exc:
        logger.error(f"Decision pipeline hatası: {exc}", exc_info=True)
        # §2.9: pipeline asla sessizce ölmez
        return {
            "summary": f"Karar üretilemedi: {exc}",
            "events": [],
            "risk": "Düşük",
            "actions": ["İnsan gözetiminde tekrar analiz et."],
            "reasoning": f"Sistem hatası: {exc}",
            "confidence": 0.0,
            "triggered_mock_tools": [],
            # Hat çöktüğünde de bu bir null yanıttır; metriklerde başarılı
            # karar olarak sayılmamalı.
            "null_response": True,
        }


async def maybe_decide(job_id: str, camera_id: str, redis_client):
    """
    Yeterli kanıt varsa karar üretir ve decision.final yayınlar.
    §2.9: VLM yokken de çalışır (timeout sonrasında).
    """
    ctx = _job_context[job_id]
    async with ctx["lock"]:
        if ctx["decided"]:
            return

        events = ctx["events"]
        if not events:
            return

        # VLM yorumu henüz gelmediyse 4 saniyeye kadar bekle
        if not ctx["vlm"]:
            try:
                await asyncio.wait_for(ctx["vlm_ready"].wait(), timeout=4.0)
            except asyncio.TimeoutError:
                logger.info(f"Job {job_id}: VLM yorumu timeout süresinde gelmedi, tek kaynakla devam ediliyor.")

        vlm = ctx["vlm"]
        events = list(ctx["events"])
        ctx["decided"] = True

    logger.info(f"Making decision for job {job_id} (events={len(events)}, vlm={len(vlm)})")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_decision_sync, events, vlm)

    # 'null_response' bayrağı _run_decision_sync tarafından set edilir; boş
    # özet kontrolü yeterli değildi (null yanıtın özeti "Bilmiyorum"dur).
    if result.get("null_response") or not result.get("summary"):
        _metrics["null_responses"] += 1
        logger.warning(
            f"Job {job_id}: null yanıt üretildi (guardrail şemayı doğrulayamadı "
            f"veya hat hata verdi) — risk={result.get('risk')!r}"
        )
    else:
        _metrics["decisions_made"] += 1

    decision = DecisionFinal(
        job_id=job_id,
        camera_id=camera_id,
        summary=result.get("summary", ""),
        events=result.get("events", []),
        risk=result.get("risk", "Düşük"),
        actions=result.get("actions", []),
        reasoning=result.get("reasoning", ""),
        confidence=float(result.get("confidence", 0.0)),
        triggered_mock_tools=result.get("triggered_mock_tools", []),
    )
    await redis_helper.publish_message(redis_client, "decision.final", decision)
    logger.info(f"Published decision.final for job {job_id} — risk={decision.risk}")

    # Temizle
    _job_context.pop(job_id, None)


async def process_message(stream_name: str, payload: dict, redis_client):
    job_id = payload.get("job_id", "")
    camera_id = payload.get("camera_id", "")

    if not job_id:
        return

    ctx = _job_context[job_id]
    ctx["camera_id"] = camera_id

    if stream_name == "event.detected":
        ctx["events"].append(payload)
        # Karar sürecini tetikle
        asyncio.create_task(maybe_decide(job_id, camera_id, redis_client))

    elif stream_name == "vlm.interpreted":
        interp = payload.get("interpretation", {})
        ctx["vlm"].append(interp)
        ctx["vlm_ready"].set()
        # VLM geldiğinde eğer henüz karar verilmediyse ve event varsa tetikle
        if ctx["events"] and not ctx["decided"]:
            asyncio.create_task(maybe_decide(job_id, camera_id, redis_client))


async def redis_consumer():
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    client = await redis_helper.get_redis_client(redis_url)

    streams = {"event.detected": ">", "vlm.interpreted": ">"}
    group_name = "decision_group"
    consumer_name = "decision_consumer_1"

    for stream in streams.keys():
        try:
            await client.xgroup_create(stream, group_name, id='0', mkstream=True)
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.warning(f"Group init error for {stream}: {e}")

    logger.info("Started Decision Service consumer...")
    while True:
        try:
            messages = await client.xreadgroup(
                groupname=group_name,
                consumername=consumer_name,
                streams=streams,
                count=10,
                block=2000,
            )
            for stream_name, msgs in messages:
                for msg_id, msg_data in msgs:
                    raw = msg_data.get("payload", "{}")
                    try:
                        payload = json.loads(raw)
                        await process_message(stream_name, payload, client)
                        await redis_helper.ack_message(client, stream_name, group_name, msg_id)
                    except Exception as e:
                        logger.error(f"Decision processing error: {e}", exc_info=True)
                        _metrics["dlq_count"] += 1
                        await redis_helper.publish_raw(client, f"{stream_name}.dlq", {
                            "msg_id": msg_id, "error": str(e)
                        })
                        await redis_helper.ack_message(client, stream_name, group_name, msg_id)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Decision consumer error: {e}")
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

app = FastAPI(title="Dalga AI Decision Service", lifespan=lifespan)
app.include_router(
    create_health_router("decision-service", get_details_fn=lambda: _metrics),
    prefix="/api/v1",
)


@app.get("/api/v1/metrics")
async def get_metrics():
    return _metrics
