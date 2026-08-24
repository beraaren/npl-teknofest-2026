"""Sözde-canlı (pseudo-live) replay motoru — kaydedilmiş analizleri canlı gibi oynatır.

ÇALIŞMA İLKESİ
--------------
Sistem canlı bir kamera duvarı gibi görünür ama çalışma anında **hiçbir model
çağrısı yapmaz**. Her video için analiz ``scripts/analyze_video_library.py`` ile
önceden üretilmiştir. Bu motor, her kamera için sanal bir oynatma kafası
(playhead) tutar ve kaydedilmiş uyarıları **olayın videodaki gerçek saniyesinde**
yayınlar. Böylece uyarı, ekranda oynayan görüntüyle eşzamanlı görünür.

Senkronizasyon nasıl kurulur
----------------------------
Motor kafanın konumunu duvar saatinden hesaplar
(``position_sec = now - cycle_started_at``) ve bunu kamera durumunda yayınlar.
Arayüz video öğesini yüklerken ``currentTime`` değerini bu konuma çeker; sunucu
ile tarayıcı aynı sanal zamanı paylaşır. Video bitince kamera yeni bir analize
geçer (dinamik replay), böylece kütüphanedeki tüm videolar sırayla ekrana gelir.

Yayınlanan akışlar
------------------
* ``decision.final`` — döngü başında bir kez: karar ajanının olay özeti, risk,
  aksiyonlar. Süpervizör kamerayı açtığında bu özeti görür.
* ``event.detected`` — her olayın tam saniyesinde.
* ``notification.push`` — ilk yüksek/kritik şiddetli olayda.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

from . import library, store

logger = logging.getLogger(__name__)

BroadcastFn = Callable[[dict], Awaitable[None]]
SaveEventFn = Callable[[str, str, dict], None]

#: Duvardaki kamera sayısı. Kütüphanedeki analiz sayısı bundan azsa kamera
#: sayısı ona indirilir; aksi hâlde aynı video birden fazla karede görünür.
DEFAULT_CAMERA_COUNT = 9

#: Şiddet -> arayüzdeki risk etiketi. Uyarı kartının rengi buna göre belirlenir.
SEVERITY_TO_RISK = {
    "critical": "Yüksek",
    "high": "Yüksek",
    "medium": "Orta",
    "low": "Düşük",
}

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}


@dataclass
class CameraStream:
    """Tek bir kameranın oynatma durumu."""

    camera_id: str
    label: str
    analysis: Optional[dict] = None
    job_id: str = ""
    cycle_started_at: float = 0.0
    cycle_index: int = 0
    #: Bu döngüde yayınlanmış son olay (arayüzde "şu an" göstergesi).
    current_event: Optional[dict] = None
    #: Bu döngüde yayınlanmış olay sayısı.
    fired_count: int = 0

    @property
    def duration_sec(self) -> float:
        if not self.analysis:
            return 0.0
        return float(self.analysis.get("video", {}).get("duration_sec") or 0.0)

    @property
    def position_sec(self) -> float:
        """Sanal oynatma kafasının konumu (saniye)."""
        if not self.analysis or not self.cycle_started_at:
            return 0.0
        elapsed = time.monotonic() - self.cycle_started_at
        return max(0.0, min(elapsed, self.duration_sec))

    def begin_cycle(self, analysis: dict) -> None:
        """Yeni bir analizle oynatma döngüsü başlatır."""
        self.analysis = analysis
        self.job_id = f"{self.camera_id}-{uuid.uuid4().hex[:8]}"
        self.cycle_started_at = time.monotonic()
        self.cycle_index += 1
        self.current_event = None
        self.fired_count = 0


class ReplayEngine:
    """Kamera duvarını sözde-canlı olarak besleyen motor."""

    def __init__(
        self,
        broadcast_fn: BroadcastFn,
        save_event_fn: SaveEventFn,
        camera_count: int = DEFAULT_CAMERA_COUNT,
    ):
        self._broadcast = broadcast_fn
        self._save_event = save_event_fn
        self._tasks: List[asyncio.Task] = []
        self._deck: List[str] = []

        pool_size = library.count()
        if pool_size == 0:
            logger.warning(
                "Analiz kütüphanesi boş. Kamera duvarı boş kalacak. "
                "Önce: python scripts/analyze_video_library.py"
            )
        # Aynı videonun iki karede birden görünmesini önlemek için kamera
        # sayısı kütüphaneyi aşmaz.
        effective = min(camera_count, pool_size) if pool_size else 0
        self.cameras: Dict[str, CameraStream] = {
            f"cam-{i:02d}": CameraStream(camera_id=f"cam-{i:02d}", label=f"Kamera {i:02d}")
            for i in range(1, effective + 1)
        }
        logger.info(
            f"ReplayEngine hazır: {len(self.cameras)} kamera, "
            f"{pool_size} analiz kütüphanede"
        )

    # -- Analiz seçimi ---------------------------------------------------

    def _on_screen_slugs(self, exclude_camera: str = "") -> set:
        """Şu anda başka kameralarda oynayan analizlerin kimliklerini döner."""
        return {
            str(cam.analysis.get("slug"))
            for cam_id, cam in self.cameras.items()
            if cam.analysis and cam_id != exclude_camera
        }

    def _draw_analysis(self, for_camera: str = "") -> Optional[dict]:
        """Desteden sıradaki analizi çeker; deste bitince yeniden karıştırır.

        Rastgele seçim yerine karıştırılmış deste kullanılır: saf rastgelelikte
        aynı video kısa aralıklarla tekrar gelir ve kütüphanenin bir kısmı hiç
        ekrana çıkmaz. Deste, her videonun sırayla görünmesini sağlar.

        Ayrıca o an başka bir karede oynayan video atlanır: kameralar farklı
        anlarda bittiği için deste yenilendiğinde aynı görüntünün iki karede
        birden çıkması mümkündü ve bu duvarda hatalı görünüyordu. Kütüphane
        kamera sayısından büyük olduğu sürece bu her zaman mümkündür.

        Args:
            for_camera: Analiz çekilen kamera (kendi videosu dışlanmaz).

        Returns:
            Seçilen analiz veya kütüphane boşsa ``None``.
        """
        on_screen = self._on_screen_slugs(exclude_camera=for_camera)
        skipped: List[str] = []
        analysis: Optional[dict] = None

        # Deste boyunca en fazla bir tur dolaş; ekranda olmayan ilk videoyu al.
        for _ in range(library.count() + 1):
            if not self._deck:
                self._deck = library.slugs()
                random.shuffle(self._deck)
                if not self._deck:
                    break

            slug = self._deck.pop()
            candidate = library.get(slug)
            if candidate is None:
                continue

            if slug in on_screen:
                # Ekranda olan videoyu şimdilik kenara ayır, desteye geri koy.
                skipped.append(slug)
                continue

            analysis = candidate
            break

        # Atlananlar desteye geri döner; sıraları kaybolmaz.
        self._deck.extend(skipped)

        if analysis is None and skipped:
            # Tüm adaylar ekranda (kütüphane kamera sayısından küçük):
            # tekrar göstermek, boş kare bırakmaktan iyidir.
            analysis = library.get(self._deck.pop())

        return analysis

    # -- Yaşam döngüsü ---------------------------------------------------

    async def start(self) -> None:
        """Her kamera için bir oynatma görevi başlatır."""
        for camera_id in self.cameras:
            task = asyncio.create_task(
                self._camera_loop(camera_id), name=f"replay-{camera_id}"
            )
            self._tasks.append(task)
        logger.info(f"ReplayEngine başlatıldı: {len(self._tasks)} kamera akışı")

    def stop(self) -> None:
        """Tüm oynatma görevlerini iptal eder."""
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

    # -- Dış arayüz ------------------------------------------------------

    def get_camera_status(self, camera_id: str) -> Optional[dict]:
        """Kameranın anlık durumunu döner (arayüz videoyu buna göre konumlar)."""
        cam = self.cameras.get(camera_id)
        if not cam or not cam.analysis:
            return None

        analysis = cam.analysis
        duration = cam.duration_sec
        position = cam.position_sec
        return {
            "job_id": cam.job_id,
            "camera_label": cam.label,
            "analysis_slug": analysis.get("slug"),
            "video_name": analysis.get("video_name"),
            "risk": analysis.get("risk"),
            "confidence": analysis.get("confidence"),
            "headline": analysis.get("headline"),
            # Karar ajanının olay özeti: süpervizör kamerayı açtığında bunu görür.
            "summary": analysis.get("summary"),
            "actions": analysis.get("actions", []),
            "duration_sec": round(duration, 2),
            # Arayüz video.currentTime değerini buraya çeker -> senkron oynatma.
            "position_sec": round(position, 2),
            "progress_percent": round((position / duration) * 100, 1) if duration else 0.0,
            "cycle_index": cam.cycle_index,
            "current_event": cam.current_event,
            "fired_count": cam.fired_count,
            "total_events": len(analysis.get("metadata", {}).get("event_timestamps", []) or []),
        }

    def current_analysis(self, camera_id: str) -> Optional[dict]:
        """Kamerada oynayan analizin tamamını döner."""
        cam = self.cameras.get(camera_id)
        return cam.analysis if cam else None

    def current_video_path(self, camera_id: str):
        """Kamerada oynayan videonun dosya yolunu döner."""
        cam = self.cameras.get(camera_id)
        if not cam or not cam.analysis:
            return None
        return library.video_path(str(cam.analysis.get("slug")))

    # -- Oynatma döngüsü -------------------------------------------------

    async def _camera_loop(self, camera_id: str) -> None:
        """Kameranın sonsuz oynatma döngüsü: analiz seç, oynat, tekrarla."""
        cam = self.cameras[camera_id]
        while True:
            analysis = self._draw_analysis(for_camera=camera_id)
            if analysis is None:
                logger.warning(f"{camera_id}: oynatılacak analiz yok, 5 sn sonra tekrar")
                await asyncio.sleep(5)
                continue

            cam.begin_cycle(analysis)
            store.save_analysis(cam.job_id, camera_id, analysis)

            await self._emit_decision(cam)
            await self._play_cycle(cam)

    async def _play_cycle(self, cam: CameraStream) -> None:
        """Bir videonun uyarılarını zamanında yayınlar, sonra süreyi tamamlar."""
        analysis = cam.analysis or {}
        stamps = sorted(
            analysis.get("metadata", {}).get("event_timestamps", []) or [],
            key=lambda e: float(e.get("seconds") or 0.0),
        )
        duration = cam.duration_sec
        notified = False
        cursor = 0.0

        for stamp in stamps:
            target = float(stamp.get("seconds") or 0.0)
            wait = target - cursor
            if wait > 0:
                await asyncio.sleep(wait)
            cursor = target

            await self._emit_event(cam, stamp)

            severity = str(stamp.get("severity") or "low")
            if not notified and SEVERITY_RANK.get(severity, 1) >= 3:
                await self._emit_notification(cam, stamp)
                notified = True

        # Videonun kalan süresini bekle: kafa sona ulaşmadan yeni döngü
        # başlarsa arayüzdeki video ile sunucunun sanal zamanı ayrışır.
        remaining = duration - cursor
        if remaining > 0:
            await asyncio.sleep(remaining)

    # -- Yayınlar --------------------------------------------------------

    async def _emit_decision(self, cam: CameraStream) -> None:
        """Döngü başında karar özetini yayınlar (süpervizörün gördüğü özet)."""
        analysis = cam.analysis or {}
        payload = {
            "job_id": cam.job_id,
            "camera_id": cam.camera_id,
            "analysis_slug": analysis.get("slug"),
            "summary": analysis.get("summary", ""),
            "headline": analysis.get("headline", ""),
            "events": analysis.get("events", []),
            "risk": analysis.get("risk", "Düşük"),
            "actions": analysis.get("actions", []),
            "reasoning": analysis.get("reasoning", ""),
            "confidence": analysis.get("confidence", 0.0),
            "triggered_mock_tools": analysis.get("triggered_mock_tools", []),
            "duration_sec": cam.duration_sec,
            "video_name": analysis.get("video_name", ""),
        }
        await self._broadcast({"stream": "decision.final", "data": payload})
        self._save_event(cam.job_id, "decision.final", payload)

    async def _emit_event(self, cam: CameraStream, stamp: dict) -> None:
        """Tek bir olay uyarısını yayınlar."""
        analysis = cam.analysis or {}
        severity = str(stamp.get("severity") or "low")
        payload = {
            "job_id": cam.job_id,
            "camera_id": cam.camera_id,
            "analysis_slug": analysis.get("slug"),
            "event_type": stamp.get("event_type", ""),
            "timestamp": stamp.get("timestamp", ""),
            "seconds": stamp.get("seconds", 0.0),
            "confidence": stamp.get("confidence", 0.0),
            "severity": severity,
            "risk": SEVERITY_TO_RISK.get(severity, "Düşük"),
            "description": stamp.get("event") or stamp.get("vlm_detail") or "",
            "video_name": analysis.get("video_name", ""),
        }
        cam.current_event = payload
        cam.fired_count += 1
        await self._broadcast({"stream": "event.detected", "data": payload})
        self._save_event(cam.job_id, "event.detected", payload)

    async def _emit_notification(self, cam: CameraStream, stamp: dict) -> None:
        """Yüksek şiddetli ilk olayda bildirim yayınlar."""
        analysis = cam.analysis or {}
        payload = {
            "job_id": cam.job_id,
            "camera_id": cam.camera_id,
            "analysis_slug": analysis.get("slug"),
            "risk": analysis.get("risk", "Orta"),
            "severity": stamp.get("severity", "high"),
            "headline": analysis.get("headline") or "Riskli durum",
            "summary": analysis.get("summary", ""),
            "actions": analysis.get("actions", []),
            "event_type": stamp.get("event_type", ""),
            "event_timestamp": stamp.get("timestamp", ""),
            "event_seconds": stamp.get("seconds", 0.0),
            "created_at": datetime.now().isoformat(),
        }
        await self._broadcast({"stream": "notification.push", "data": payload})
        self._save_event(cam.job_id, "notification.push", payload)
