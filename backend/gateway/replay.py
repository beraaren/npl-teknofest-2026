"""Pseudolive replay engine — UI/demo katmanı için 9 kamerayı sürekli döndürür.

Her kamera bağımsız bir asyncio task'ı olarak çalışır; analiz JSON'larından
rastgele seçim yapar, decision.final / event.detected / notification.push
olaylarını WebSocket üzerinden yayınlar ve SQLite events tablosuna yazar.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from . import store

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ANALYSES_DIR = PROJECT_ROOT / "data" / "pseudolive" / "analyses"
VIDEOS_DIR = PROJECT_ROOT / "data" / "pseudolive" / "videos"
CAMERA_IDS = [f"cam-{i:02d}" for i in range(1, 10)]


BroadcastFn = Callable[[dict], Awaitable[None]]
SaveEventFn = Callable[[str, str, dict], None]


class ReplayEngine:
    """Sürekli pseudolive analiz akışı üreten motor."""

    def __init__(
        self,
        broadcast_fn: BroadcastFn,
        save_event_fn: SaveEventFn,
        analyses_dir: Path | None = None,
    ):
        self._broadcast = broadcast_fn
        self._save_event = save_event_fn
        self._analyses_dir = analyses_dir or ANALYSES_DIR
        self._pool: List[dict] = []
        self._states: Dict[str, dict] = {cam: {} for cam in CAMERA_IDS}
        self._tasks: List[asyncio.Task] = []
        self._load_pool()

    def _load_pool(self) -> None:
        if not self._analyses_dir.exists():
            logger.warning(f"Pseudolive analyses directory not found: {self._analyses_dir}")
            return
        for path in sorted(self._analyses_dir.glob("*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "metadata" in data:
                    self._pool.append(data)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(f"Failed to load {path}: {exc}")
        logger.info(f"Loaded {len(self._pool)} pseudolive analyses from {self._analyses_dir}")

    async def start(self) -> None:
        """Her kamera için bir replay task başlatır."""
        for camera_id in CAMERA_IDS:
            task = asyncio.create_task(
                self._camera_loop(camera_id),
                name=f"replay-{camera_id}",
            )
            self._tasks.append(task)
        logger.info("ReplayEngine started for 9 cameras")

    def stop(self) -> None:
        """Tüm replay task'larını iptal eder."""
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

    def get_camera_status(self, camera_id: str) -> dict | None:
        """Aktif analizin metadata özetini döner."""
        state = self._states.get(camera_id)
        if not state:
            return None
        metadata = state.get("metadata") or {}
        analysis = state.get("analysis") or {}
        started_at = state.get("started_at", 0.0)
        duration_sec = float(metadata.get("duration_sec", 0) or 0)
        elapsed = time.monotonic() - started_at
        progress = 0.0
        if duration_sec > 0:
            progress = min(100.0, round((elapsed / duration_sec) * 100, 1))
        return {
            "job_id": state.get("job_id"),
            "camera_label": metadata.get("camera_label"),
            "risk": analysis.get("risk"),
            "video_file": metadata.get("video_file"),
            "duration_sec": duration_sec,
            "current_event": state.get("current_event"),
            "progress_percent": progress,
        }

    def current_video_path(self, camera_id: str) -> Path | None:
        """Aktif analizin video dosyasının yolunu döner (varsa)."""
        state = self._states.get(camera_id)
        if not state:
            return None
        metadata = state.get("metadata") or {}
        video_file = metadata.get("video_file")
        if not video_file:
            return None
        path = VIDEOS_DIR / video_file
        return path if path.exists() else None

    async def _camera_loop(self, camera_id: str) -> None:
        while True:
            if not self._pool:
                logger.warning(f"No pseudolive analyses available for {camera_id}; retrying in 5s")
                await asyncio.sleep(5)
                continue

            analysis = random.choice(self._pool)
            job_id = f"{camera_id}-{uuid.uuid4().hex[:8]}"
            metadata = analysis.get("metadata", {})
            duration_sec = float(metadata.get("duration_sec", 30) or 30)

            store.save_analysis(job_id, camera_id, analysis)
            self._states[camera_id] = {
                "job_id": job_id,
                "analysis": analysis,
                "metadata": metadata,
                "started_at": time.monotonic(),
                "current_event": None,
            }

            decision_payload = {
                "job_id": job_id,
                "camera_id": camera_id,
                "summary": analysis.get("summary", ""),
                "events": analysis.get("events", []),
                "risk": analysis.get("risk", "Düşük"),
                "actions": analysis.get("actions", []),
                "reasoning": analysis.get("reasoning", ""),
                "confidence": analysis.get("confidence", 0.0),
                "triggered_mock_tools": analysis.get("triggered_mock_tools", []),
            }
            await self._broadcast({"stream": "decision.final", "data": decision_payload})
            self._save_event(job_id, "decision.final", decision_payload)

            await self._play_analysis(camera_id, job_id, analysis, duration_sec)

    async def _play_analysis(
        self,
        camera_id: str,
        job_id: str,
        analysis: dict,
        duration_sec: float,
    ) -> None:
        metadata = analysis.get("metadata", {})
        event_timestamps = sorted(
            metadata.get("event_timestamps", []),
            key=lambda e: float(e.get("seconds", 0)),
        )
        risk = analysis.get("risk", "Düşük")

        # Zaman çizelgesi oluştur: (saniye, tür, veri)
        timeline: List[tuple[float, str, Any]] = []
        for ev in event_timestamps:
            timeline.append((float(ev.get("seconds", 0)), "event", ev))

        if risk in ("Yüksek", "Orta"):
            notify_at = self._pick_notification_time(analysis, duration_sec)
            timeline.append((notify_at, "notification", None))

        timeline.sort(key=lambda x: x[0])

        current_time = 0.0
        for target_time, kind, data in timeline:
            sleep_time = max(0.0, target_time - current_time)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            current_time = target_time

            if kind == "event":
                await self._emit_event(camera_id, job_id, data, analysis)
            elif kind == "notification":
                await self._emit_notification(camera_id, job_id, analysis)

        remaining = max(0.0, duration_sec - current_time)
        if remaining > 0:
            await asyncio.sleep(remaining)

    async def _emit_event(
        self,
        camera_id: str,
        job_id: str,
        event_ts: dict,
        analysis: dict,
    ) -> None:
        event_type = event_ts.get("event_type", "")
        description, confidence = self._describe_event(event_type, analysis)
        payload = {
            "job_id": job_id,
            "camera_id": camera_id,
            "event_type": event_type,
            "timestamp": event_ts.get("timestamp", ""),
            "confidence": confidence,
            "description": description,
        }
        self._states[camera_id]["current_event"] = payload
        await self._broadcast({"stream": "event.detected", "data": payload})
        self._save_event(job_id, "event.detected", payload)

    def _describe_event(self, event_type: str, analysis: dict) -> tuple[str, float]:
        metadata = analysis.get("metadata", {})
        for signal in metadata.get("geometric_signals", []):
            if signal.get("event_type") == event_type:
                return signal.get("description", ""), signal.get("confidence", 0.0)
        for ev in analysis.get("events", []):
            if ev.get("event_type") == event_type:
                return ev.get("event", ""), ev.get("confidence", 0.0)
        return "", 0.0

    async def _emit_notification(
        self,
        camera_id: str,
        job_id: str,
        analysis: dict,
    ) -> None:
        risk = analysis.get("risk", "Düşük")
        summary = analysis.get("summary", "")
        payload = {
            "job_id": job_id,
            "camera_id": camera_id,
            "risk": risk,
            "headline": summary[:100] or f"{risk} risk uyarısı",
            "summary": summary,
            "actions": analysis.get("actions", []),
            "created_at": datetime.now().isoformat(),
        }
        await self._broadcast({"stream": "notification.push", "data": payload})
        self._save_event(job_id, "notification.push", payload)

    def _pick_notification_time(self, analysis: dict, duration_sec: float) -> float:
        risk = analysis.get("risk", "Düşük")
        segments = analysis.get("metadata", {}).get("risk_segments", [])
        if risk == "Yüksek":
            for seg in segments:
                if seg.get("risk") == "Yüksek":
                    return float(seg.get("start_sec", 5))
        # Orta veya Yüksek segment bulunamazsa ilk segmenti kullan
        if segments:
            return float(segments[0].get("start_sec", 5))
        return min(5.0, duration_sec)
