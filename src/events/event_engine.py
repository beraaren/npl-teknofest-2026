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
        # Sahne grafiği `near` kenarının kurulup kurulmayacağına karar veren TABAN
        # eşik. Bu değerin altında kalan çiftler için near kenarı hiç oluşmaz ve
        # _rule_proximity o çifti göremez — bu yüzden gerçek karar (sinyal üretilsin
        # mi) burada değil, _rule_proximity içinde merkezler arası gerçek mesafe
        # üzerinden verilir. Buradaki değer sadece "bu eşiğin altındaki adayları
        # kenar listesine dahil et" filtresidir; her karede
        # `_compute_proximity_graph_threshold` ile ölçeğe göre büyütülür.
        self.proximity_threshold = float(
            config.thresholds.proximity.get("distance_threshold_pixels", DEFAULT_PROXIMITY_THRESHOLD)
        )
        self._proximity_dangerous_classes = {
            cls for pair in config.thresholds.proximity.get("dangerous_pairs", [["arac", "insan"]]) for cls in pair
        }
        rule_thresholds = {
            "enabled_rules": config.enabled_rules,
            **config.thresholds.model_dump(),
        }
        self.rules = RuleSet(rule_thresholds, fps=fps)
        # TrackStateMachine, RuleSet ile AYNI thresholds sözlüğünü kullanır
        # (isimle okuma deseni — bkz. TrackStateMachine docstring'i). Yeni bir
        # kural saniye cinsinden bir pencere ayarı gerektirdiğinde, bu iki
        # sınıfın constructor imzasına dokunmadan sadece config.yaml'a ve
        # ilgili `_rule_*`/`TrackState.update` okuma satırına eklenir.
        self.states = TrackStateMachine(fps=fps, thresholds=rule_thresholds)
        self.signals: List[EventSignal] = []
        # Kalıcı track kayıtları: track_id -> TrackedObject. Bu sözlük olmadan
        # her process_observation() çağrısında history=[det] ile sıfırdan bir
        # TrackedObject üretilir; TrackedObject.speed en az 2 geçmiş kaydı
        # gerektirdiğinden hız her zaman (0.0, 0.0) döner ve buna bağlı tüm
        # kurallar (örn. person_fall) hiçbir zaman tetiklenemez. Bu sözlük,
        # ObserverAgent.observe_frame() içindeki self.tracks ile aynı deseni
        # kullanarak track geçmişinin kareler arasında korunmasını sağlar.
        self._tracked_objects: Dict[int, TrackedObject] = {}
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
            proximity_threshold=self._compute_proximity_graph_threshold(tracks),
        )

        new_signals = self.rules.evaluate(tracks, self.states, graph)
        # Yinelenen sinyalleri önle (aynı track + event_type son 10 saniye içinde varsa atla)
        filtered = []
        for sig in new_signals:
            if not self._is_recent(sig):
                filtered.append(sig)
                self.signals.append(sig)
        return filtered

    def _compute_proximity_graph_threshold(self, tracks: List[TrackedObject]) -> float:
        """Sahne grafiğinin `near` kenarı için o kareye özel eşiği hesaplar.

        `SceneGraph.build_relations()` sabit bir eşik alır: bu eşiğin üstünde
        kalan çiftler için `near` kenarı hiç kurulmaz, dolayısıyla
        `_rule_proximity` o çifti asla göremez (kural içindeki oranlı/ölçekli
        mantık kenarın var olmasına bağlıdır). Sabit `distance_threshold_pixels`
        (örn. 100px) kameraya yakın çekilmiş sahnelerde çok düşük kalabilir:
        bbox yüksekliği 300-400px olan bir forklift-insan çiftinde, gerçek
        tehlikeli mesafe (kendi ölçeklerinin ~1 katı, örn. 270-300px) sabit
        eşiğin üstünde kalır ve near kenarı hiç oluşmaz (bkz. proximity.mp4
        gözlemi: forklift 278px'e kadar yaklaştı, hep "güvenli" sayıldı).

        Bu yüzden buradaki eşik, o karede görülen tehlikeli sınıflardaki
        (`dangerous_pairs`) track'lerin en büyük `scale_ema`'sına göre
        `distance_threshold_ratio` oranında büyütülür. Nihai kabul/ret kararı
        hâlâ `_rule_proximity`'de verilir (`estimated_dist <= effective_limit`);
        burada yalnızca kenarın kurulabilmesi için yeterli genişlikte bir taban
        sağlanır — kenar kurulmazsa kural hiç çalışamaz.

        Args:
            tracks: Bu karedeki aktif takip nesneleri.

        Returns:
            `near` kenarı için kullanılacak mesafe eşiği (piksel). En az
            yapılandırılan sabit `distance_threshold_pixels` kadardır.
        """
        proximity_cfg = self.config.thresholds.proximity
        distance_ratio = proximity_cfg.get("distance_threshold_ratio", 1.0)

        max_scale = 0.0
        for t in tracks:
            if t.class_name not in self._proximity_dangerous_classes:
                continue
            state = self.states.get(t.track_id)
            if state and state.scale_ema > max_scale:
                max_scale = state.scale_ema

        if max_scale <= 0.0:
            return self.proximity_threshold

        return max(self.proximity_threshold, max_scale * distance_ratio)

    def _observation_to_tracks(self, observation: Dict[str, Any]) -> List[TrackedObject]:
        """Gözlem sözlüğündeki ham track ve detection verilerini `TrackedObject` nesnelerine dönüştürür.

        Aynı `track_id` için tekrar çağrıldığında **var olan** `TrackedObject`
        güncellenir (geçmişe yeni bir `Detection` eklenir); sıfırdan
        yeniden yaratılmaz. Bu, `TrackedObject.speed`'in en az iki geçmiş
        kaydına ihtiyaç duyması ve geçmişin kareler arasında korunması
        gerekmesi nedeniyle zorunludur — aksi halde hız her zaman
        `(0.0, 0.0)` hesaplanır ve buna dayanan kurallar (örn. `person_fall`)
        hiçbir zaman tetiklenemez.

        Args:
            observation (Dict[str, Any]): Kare bazlı gözlem verisi.

        Returns:
            List[TrackedObject]: Takip nesneleri listesi (bu kareden itibaren
            aktif olanlar; kalıcı kayıtlar `self._tracked_objects` içindedir).
        """
        from ..perception.detector import Detection
        tracks: List[TrackedObject] = []
        for t in observation.get("tracks", []):
            tid = t["track_id"]
            # Daha zengin detection bilgisi varsa kullan (önce track_id ile, yoksa class ile eşle)
            det_data = next(
                (d for d in observation.get("detections", []) if d.get("track_id") == tid),
                next((d for d in observation.get("detections", []) if d.get("class") == t["class"]), None),
            )
            if det_data:
                det = Detection(
                    class_name=det_data["class"],
                    confidence=det_data.get("confidence", 1.0),
                    bbox=tuple(det_data.get("bbox", [0, 0, 0, 0])),
                    frame_idx=det_data.get("frame_idx", 0),
                    track_id=tid,
                )
            else:
                det = Detection(
                    class_name=t["class"],
                    confidence=1.0,
                    bbox=(0.0, 0.0, 0.0, 0.0),
                    track_id=tid,
                )

            existing = self._tracked_objects.get(tid)
            if existing is not None:
                existing.update(det)
            else:
                existing = TrackedObject(track_id=tid, class_name=t["class"], initial_detection=det)
                self._tracked_objects[tid] = existing
            tracks.append(existing)
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
