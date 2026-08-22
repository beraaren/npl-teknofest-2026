"""api-gateway — dış dünyanın tek giriş noktası.

§2.3.6 ve §2.5 tasarımına uygun:
  - REST: cameras + analyses yönetimi (prefix /api/v1)
  - Olay köprüsü: Redis akışlarını dinler, decision.final SQLite'a yazar,
    event.detected / tool.executed / notification.push olayları events tablosuna kaydedilir
  - WebSocket /ws: tüm olayları bağlı istemcilere iter, ?camera_id= filtresi
  - GET /api/v1/metrics: KPI'lar
  - GET /health: gateway sağlık (prefix olmadan da erişilebilir)
"""
import asyncio
import json
import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routers import cameras, analyses, pseudolive, ops
from ..common.health import create_health_router
from ..common import redis as redis_helper
from . import store
from .replay import ReplayEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# §2.10: KPI sayaçları
_metrics = {
    "events_detected": 0,
    "decisions_made": 0,
    "tools_executed": 0,
    "notifications_sent": 0,
    "risk_distribution": {"Düşük": 0, "Orta": 0, "Yüksek": 0},
}


class ConnectionManager:
    """WebSocket bağlantı yöneticisi — §2.5: ?camera_id= filtresi destekler."""

    def __init__(self):
        # {websocket: camera_id_filter}  (None → tüm olaylar)
        self.connections: dict[WebSocket, str | None] = {}

    async def connect(self, websocket: WebSocket, camera_id: str | None = None):
        await websocket.accept()
        self.connections[websocket] = camera_id
        logger.info(f"WS client connected (filter={camera_id})")

    def disconnect(self, websocket: WebSocket):
        self.connections.pop(websocket, None)

    async def broadcast(self, message: dict):
        """Mesajı, camera_id filtresiyle eşleşen tüm istemcilere gönderir.

        Aynı zamanda modül düzeyindeki KPI sayaçlarını günceller; böylece
        hem Redis üzerinden gelen gerçek olaylar hem de replay motoru
        ürettikleri pseudo-live olaylar metriklere yansır.
        """
        msg_camera = message.get("data", {}).get("camera_id")
        dead = []
        for ws, cam_filter in list(self.connections.items()):
            # Filtre yoksa veya filtre eşleşiyorsa gönder
            if cam_filter is None or cam_filter == msg_camera:
                try:
                    await ws.send_json(message)
                except Exception:
                    dead.append(ws)
        for ws in dead:
            self.connections.pop(ws, None)

        # KPI güncelleme (hem Redis hem replay için tek yerden)
        stream_name = message.get("stream")
        if stream_name == "event.detected":
            _metrics["events_detected"] += 1
        elif stream_name == "decision.final":
            _metrics["decisions_made"] += 1
            risk = message.get("data", {}).get("risk", "")
            if risk in _metrics["risk_distribution"]:
                _metrics["risk_distribution"][risk] += 1
        elif stream_name == "tool.executed":
            _metrics["tools_executed"] += 1
        elif stream_name == "notification.push":
            _metrics["notifications_sent"] += 1


manager = ConnectionManager()


async def redis_listener():
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    client = await redis_helper.get_redis_client(redis_url)

    streams = {
        "decision.final": ">",
        "event.detected": ">",
        "tool.executed": ">",
        "notification.push": ">",
    }
    group_name = "gateway_group"
    consumer_name = "gateway_consumer_1"

    for stream in streams.keys():
        try:
            await client.xgroup_create(stream, group_name, id='0', mkstream=True)
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.warning(f"Group init error for {stream}: {e}")

    logger.info("Started Redis listener bridge...")
    while True:
        try:
            messages = await client.xreadgroup(
                groupname=group_name,
                consumername=consumer_name,
                streams=streams,
                count=10,
                block=1000,
            )
            for stream_name, msgs in messages:
                for msg_id, msg_data in msgs:
                    try:
                        payload = json.loads(msg_data.get("payload", "{}"))
                    except json.JSONDecodeError:
                        logger.error(f"Gateway: JSON parse error in {stream_name}")
                        await client.xack(stream_name, group_name, msg_id)
                        continue

                    job_id = payload.get("job_id", "")

                    # decision.final → SQLite analyses
                    if stream_name == "decision.final":
                        camera_id = payload.get("camera_id", "")
                        store.save_analysis(job_id, camera_id, payload)

                    # Tüm olayları events tablosuna kaydet (debug/demo)
                    if job_id and stream_name in ("event.detected", "tool.executed",
                                                   "notification.push", "decision.final"):
                        store.save_event(job_id, stream_name, payload)

                    # WebSocket'e ilet (broadcast KPI'ları da günceller)
                    await manager.broadcast({"stream": stream_name, "data": payload})
                    await client.xack(stream_name, group_name, msg_id)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Redis listener error: {e}")
            await asyncio.sleep(1)

    await client.aclose()


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    store.seed_cameras()

    replay_engine = ReplayEngine(
        broadcast_fn=manager.broadcast,
        save_event_fn=store.save_event,
    )
    await replay_engine.start()
    app.state.replay_engine = replay_engine
    app.state.broadcast_fn = manager.broadcast

    task = asyncio.create_task(redis_listener())
    yield
    replay_engine.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title="Dalga AI Gateway",
    description="API-Gateway: REST + WebSocket + Redis→SQLite köprüsü",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cameras.router, prefix="/api/v1")
app.include_router(analyses.router, prefix="/api/v1")
app.include_router(pseudolive.router, prefix="/api/v1")
app.include_router(ops.router, prefix="/api/v1")
import httpx

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_STATIC_DIR.mkdir(parents=True, exist_ok=True)

DOWNSTREAM_SERVICES = {
    "camera-ingest": os.environ.get("INGEST_SERVICE_URL", "http://camera-ingest:8001"),
    "perception": os.environ.get("PERCEPTION_SERVICE_URL", "http://perception:8002"),
    "vlm": os.environ.get("VLM_SERVICE_URL", "http://vlm:8003"),
    "decision": os.environ.get("DECISION_SERVICE_URL", "http://decision:8004"),
    "notification": os.environ.get("NOTIFICATION_SERVICE_URL", "http://notification:8005"),
}


async def check_downstream_services() -> dict:
    """Downstream mikroservislerin sağlık durumlarını paralel sorgular."""
    results = {}
    async with httpx.AsyncClient(timeout=1.5) as client:
        for name, base_url in DOWNSTREAM_SERVICES.items():
            try:
                resp = await client.get(f"{base_url}/api/v1/health")
                if resp.status_code == 200:
                    results[name] = resp.json()
                else:
                    results[name] = {"status": "error", "code": resp.status_code}
            except Exception as e:
                results[name] = {"status": "unreachable", "error": str(e)}
    return results


@app.get("/health")
@app.get("/api/v1/health")
async def health_root():
    """§2.3.6 ve §2.5: Gateway + downstream servis sağlık özeti."""
    downstream = await check_downstream_services()
    any_down = any(v.get("status") not in ("ok", "healthy") for v in downstream.values())
    status = "degraded" if any_down else "ok"
    return {
        "status": status,
        "service": "api-gateway",
        "downstream": downstream,
    }


@app.get("/api/v1/metrics")
async def get_metrics():
    """§2.5 ve §2.10: KPI endpoint."""
    return _metrics


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, camera_id: str | None = Query(default=None)):
    """§2.5: ?camera_id= filtresi destekler."""
    await manager.connect(websocket, camera_id=camera_id)
    try:
        while True:
            # İstemciden gelen mesajları dinle (bağlantıyı canlı tut)
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# UI statik dosyalarını sun (html=True ile / → index.html)
app.mount(
    "/",
    StaticFiles(directory=_STATIC_DIR, html=True),
    name="static",
)
