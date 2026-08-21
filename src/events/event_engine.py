"""Olay Tespit Motoru (Event Engine) Modülü.

Bu modül, Kanal A algılama katmanından gelen gözlemleri (`observation`) alır;
nesne durumlarını (`TrackStateMachine`) günceller, anlık sahne grafiğini (`SceneGraph`)
oluşturur ve geometrik kuralları (`RuleSet`) işleterek olay sinyalleri (`EventSignal`) üretir.
Aynı olayın kısa aralıklarla tekrar etmesini (spam) önleyen zaman filtresine sahiptir.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ..config import EventsConfig
from ..perception.scene_graph import DEFAULT_PROXIMITY_THRESHOLD, SceneGraph
from ..perception.tracker import TrackedObject
from ..utils.logger import get_logger
from .rules import EventSignal, RuleSet
from .state_machine import TrackStateMachine


class EventEngine:
    """Durum makinesi ve kural setini birleştiren ana olay motoru.

    Attributes:
        config (EventsConfig): Olay motoru konfigürasyon nesnesi.
        fps (float): Zamanlama hesaplamaları için kullanılan kare hızı.
        proximity_threshold (float): Sahne grafiği `near` kenarları için kullanılan
            merkez mesafesi eşiği (piksel). `config.yaml` içindeki
            `events.thresholds.proximity.distance_threshold_pixels` değerinden okunur.
        rules (RuleSet): Aktif geometrik kurallar motoru.
        states (TrackStateMachine): Takip edilen nesnelerin durum makinesi (durağanlık, düşme, devrilme sayaçları).
        signals (List[EventSignal]): Video boyunca üretilen tüm tekil olay sinyalleri listesi.
        logger: Modül loglama aracı.
    """

    def __init__(self, config: EventsConfig, fps: float = 25.0):
        """EventEngine nesnesini başlatır.

        Args:
            config (EventsConfig): Kurallar ve eşik değerlerini içeren yapılandırma.
            fps (float): Videonun işlenme hızı (Kanal A efektif FPS'i).
        """
        self.config = config
        self.fps = fps
        # Sahne grafiği kenarları ile _rule_proximity aynı eşiği kullanmak zorundadır:
        # kenar ağırlığı (1 - dist/threshold) ile kuralın mesafe geri hesabı
        # ((1 - weight) * threshold) ancak aynı tabanla tutarlı sonuç verir.
        self.proximity_threshold = float(
            config.thresholds.proximity.get("distance_threshold_pixels", DEFAULT_PROXIMITY_THRESHOLD)
        )
        self.rules = RuleSet({
            "enabled_rules": config.enabled_rules,
            **config.thresholds.model_dump(),
        }, fps=fps)
        self.states = TrackStateMachine(fps=fps)
        self.signals: List[EventSignal] = []
        self.logger = get_logger("EventEngine")

    def process_observation(self, observation: Dict[str, Any]) -> List[EventSignal]:
        """Tek bir video karesine ait gözlemi işleyerek yeni olay sinyalleri üretir.

        İşlem Adımları:
          1. Gözlemdeki track verilerini `TrackedObject` formatına çevirir ve durum makinesini günceller.
          2. Gözlemdeki sahne grafiği sözlüğünü `SceneGraph.from_dict()` ile geri kurar;
             ilişkiler yapılandırmadan gelen `proximity_threshold` ile yeniden hesaplanır.
          3. `RuleSet.evaluate()` ile kuralları çalıştırır.
          4. `_is_recent()` filtresi ile son 10 saniye içinde aynı nesne için aynı olay üretilmişse yineleneni eler.

        Args:
            observation (Dict[str, Any]): `ObserverAgent.observe_frame()` çıktısı olan sözlük.

        Returns:
            List[EventSignal]: Bu karede ilk kez tetiklenen yeni olay sinyalleri listesi.
        """
        tracks = self._observation_to_tracks(observation)
        self.states.update(tracks)

        graph = SceneGraph.from_dict(
            observation.get("scene_graph", {}),
            proximity_threshold=self.proximity_threshold,
        )

        new_signals = self.rules.evaluate(tracks, self.states, graph)
        # Yinelenen sinyalleri önle (aynı track + event_type son 10 saniye içinde varsa atla)
        filtered = []
        for sig in new_signals:
            if not self._is_recent(sig):
                filtered.append(sig)
                self.signals.append(sig)
        return filtered

    def _observation_to_tracks(self, observation: Dict[str, Any]) -> List[TrackedObject]:
        """Gözlem sözlüğündeki ham track ve detection verilerini `TrackedObject` nesnelerine dönüştürür.

        Args:
            observation (Dict[str, Any]): Kare bazlı gözlem verisi.

        Returns:
            List[TrackedObject]: Takip nesneleri listesi.
        """
        from ..perception.detector import Detection
        tracks: List[TrackedObject] = []
        for t in observation.get("tracks", []):
            det = Detection(
                class_name=t["class"],
                confidence=1.0,
                bbox=(0.0, 0.0, 0.0, 0.0),
            )
            # Daha zengin detection bilgisi varsa kullan (önce track_id ile, yoksa class ile eşle)
            det_data = next(
                (d for d in observation.get("detections", []) if d.get("track_id") == t["track_id"]),
                next((d for d in observation.get("detections", []) if d.get("class") == t["class"]), None),
            )
            if det_data:
                det = Detection(
                    class_name=det_data["class"],
                    confidence=det_data.get("confidence", 1.0),
                    bbox=tuple(det_data.get("bbox", [0, 0, 0, 0])),
                    frame_idx=det_data.get("frame_idx", 0),
                    track_id=t["track_id"],
                )
            to = TrackedObject(track_id=t["track_id"], class_name=t["class"], initial_detection=det)
            to.history = [det]
            tracks.append(to)
        return tracks

    def _is_recent(self, sig: EventSignal, window_seconds: float = 10.0) -> bool:
        """Aynı tür olayın belirli bir süre içinde yinelenip yinelenmediğini kontrol eder.

        Örnek: Forklift 5 saniye boyunca devrik kaldıysa her karede ayrı bir 'devrildi'
        sinyali üretmek yerine tek bir sinyal üretilmesini sağlar (de-duplication).

        Args:
            sig (EventSignal): Kontrol edilecek yeni olay sinyali.
            window_seconds (float): Yinelenme engelleme zaman penceresi (varsayılan 10.0 saniye).

        Returns:
            bool: Son `window_seconds` içinde aynı nesnelerle aynı olay varsa True, aksi halde False.
        """
        for prev in reversed(self.signals):
            if prev.event_type == sig.event_type and set(prev.involved_track_ids) == set(sig.involved_track_ids):
                if abs(prev.timestamp - sig.timestamp) <= window_seconds:
                    return True
        return False

    def get_signals(self) -> List[Dict[str, Any]]:
        """Video boyunca toplanan tüm olay sinyallerini JSON formatında döner.

        Returns:
            List[Dict[str, Any]]: 'MM:SS' zaman damgalı olay sözlükleri listesi.
        """
        return [s.to_dict() for s in self.signals]
