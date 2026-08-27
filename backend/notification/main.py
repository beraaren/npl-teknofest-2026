"""notification-service — decision.final tüketip araçları çalıştırır, UI'a bildirim iter.

§2.3.5 tasarımına uygun:
  - decision.final geldiğinde:
    1. Model araç seçtiyse onları çalıştır (triggered_mock_tools)
    2. Seçmediyse suggest_tools ile kural tabanlı öneri çalıştır
    3. Her araç sonucunu tool.executed olarak yayınla
    4. notification.push ile operatöre Türkçe bildirim gönder
  - Şartnamedeki "mock fonksiyonların ajanın araçları olarak kullanılması" kriteri
    burada görünür ve izlenebilir olur
"""
import asyncio
import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI
from contextlib import asynccontextmanager

from ..common.health import create_health_router
from ..common import redis as redis_helper
from ..contracts.messages import DecisionFinal, ToolExecuted, NotificationPush

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrik sayaçları
_metrics = {"notifications_sent": 0, "tools_executed": 0, "decisions_processed": 0, "dlq_count": 0}


def _get_mock_registry():
    """MockToolRegistry nesnesini yükler; başarısız olursa None döner."""
    try:
        import sys
        _bera_root = str(Path(__file__).resolve().parent.parent.parent)
        if _bera_root not in sys.path:
            sys.path.insert(0, _bera_root)
        from src.reasoning.mock_tools import MockToolRegistry
        return MockToolRegistry()
    except Exception as exc:
        logger.warning(f"MockToolRegistry yüklenemedi: {exc}")
        return None


async def process_decision(decision_data: dict, redis_client):
    """
    decision.final işler:
    1. Araçları çalıştır (MockToolRegistry)
    2. tool.executed yayınla
    3. notification.push yayınla
    """
    decision = DecisionFinal(**decision_data)
    logger.info(f"Processing decision for job {decision.job_id} — risk={decision.risk}")

    registry = _get_mock_registry()
    tools_to_run = []

    # §2.3.5: Model araç seçtiyse → onları çalıştır
    if decision.triggered_mock_tools:
        for tool_call in decision.triggered_mock_tools:
            name = tool_call.get("tool_name") or tool_call.get("tool", "")
            params = tool_call.get("params", {})
            if name:
                tools_to_run.append({"tool_name": name, "params": params})
    # Model araç seçmediyse → kural tabanlı öneri
    elif registry:
        suggested = registry.suggest_tools_for_results(decision.results)
        tools_to_run = suggested
        logger.info(
            f"Model araç seçmedi; kural tabanlı öneri: "
            f"{[t['tool_name'] for t in suggested]}"
        )

    # Her aracı çalıştır ve tool.executed yayınla
    executed_tools = []
    for tool_item in tools_to_run:
        tool_name = tool_item.get("tool_name", "")
        params = tool_item.get("params", {})

        if registry:
            result = registry.execute(tool_name, params)
            status = result.get("status", "success")
            mock_result = result.get("mock_result", f"{tool_name} çalıştırıldı.")
        else:
            # Registry yüklenemedi → minimal fallback (loglama yeterli)
            status = "success"
            mock_result = f"{tool_name} simüle edildi (registry yüklü değil)."

        logger.info(f"Tool executed: {tool_name} | status={status}")

        executed = ToolExecuted(
            job_id=decision.job_id,
            tool_name=tool_name,
            params=params,
            status=status,
            mock_result=mock_result,
        )
        await redis_helper.publish_message(redis_client, "tool.executed", executed)
        executed_tools.append({"tool": tool_name, "status": status})
        _metrics["tools_executed"] += 1

    # §2.3.5: notification.push — "00:15 Forklift devrildi" formatında headline
    first_event = next(
        (result for result in decision.results
         if result.get("result_type") == "incident" and not result.get("uncertain")),
        decision.events[0] if decision.events else {},
    )
    time_str = first_event.get("time", "00:00")
    event_desc = first_event.get("event", first_event.get("event_type", "Olay tespit edildi"))
    headline = f"{time_str} {event_desc}"

    notification = NotificationPush(
        job_id=decision.job_id,
        camera_id=decision.camera_id,
        risk={"critical": "Yüksek", "high": "Yüksek", "medium": "Orta", "low": "Düşük"}.get(
            str(first_event.get("severity") or "unknown"), "Düşük"
        ),
        headline=headline,
        summary=decision.summary,
        actions=decision.actions,
        created_at=datetime.now(timezone.utc),
    )
    await redis_helper.publish_message(redis_client, "notification.push", notification)
    _metrics["notifications_sent"] += 1
    _metrics["decisions_processed"] += 1

    logger.info(
        f"Notification sent for job {decision.job_id}: '{headline}' "
        f"| tools={[t['tool'] for t in executed_tools]}"
    )


async def redis_consumer():
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    client = await redis_helper.get_redis_client(redis_url)

    stream_name = "decision.final"
    group_name = "notification_group"
    consumer_name = "notification_consumer_1"

    try:
        await client.xgroup_create(stream_name, group_name, id='0', mkstream=True)
    except Exception as e:
        if "BUSYGROUP" not in str(e):
            logger.warning(f"Group init error for {stream_name}: {e}")

    logger.info("Started Notification Service consumer...")
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
                        await process_decision(payload, client)
                        await redis_helper.ack_message(client, stream_name, group_name, msg_id)
                    except Exception as e:
                        logger.error(f"Notification processing error: {e}", exc_info=True)
                        _metrics["dlq_count"] += 1
                        await redis_helper.publish_raw(client, f"{stream_name}.dlq", {
                            "msg_id": msg_id, "error": str(e)
                        })
                        await redis_helper.ack_message(client, stream_name, group_name, msg_id)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Notification consumer error: {e}")
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

app = FastAPI(title="Dalga AI Notification Service", lifespan=lifespan)
app.include_router(
    create_health_router("notification-service", get_details_fn=lambda: _metrics),
    prefix="/api/v1",
)


@app.get("/api/v1/metrics")
async def get_metrics():
    return _metrics
