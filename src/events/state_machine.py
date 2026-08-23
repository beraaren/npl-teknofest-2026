"""Track bazlı durum makinesi."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..perception.tracker import TrackedObject


#: Ölçek (scale_ema) üstel ortalamasının ağırlığı. Yüksek değer = yeni kareye
#: daha çok güven (daha hızlı adapte olur, gürültüye daha açık); düşük değer =
#: daha yavaş adapte olur, tek karelik tespit gürültüsüne (occlusion, kısmi
#: bbox) karşı daha dayanıklı. 0.2 hem tepkisel hem stabil bir orta nokta.
SCALE_EMA_ALPHA: float = 0.2


@dataclass
class TrackState:
    """Bir track_id için kareler arasında korunan türetilmiş (enrichment) durum.

    Bu sınıf, birden fazla kuralın (fall, proximity, gathering) tekrar tekrar
    aynı hesabı yapmasını önlemek için "ölçek" (scale_ema) bilgisini merkezi
    olarak bir kez hesaplar ve saklar. Ölçek, nesnenin bbox yüksekliğinin
    üstel hareketli ortalamasıdır (piksel) ve kameraya uzaklığın (derinliğin)
    dolaylı bir göstergesidir: kameraya yakın nesnelerde büyük, uzak
    nesnelerde küçüktür. Kurallar sabit piksel eşiği yerine bu değere oranlı
    eşikler kullanarak kameraya uzaklıktan bağımsız (scale-invariant) çalışır
    — aynı `src/perception/scene_graph.py` içindeki `wearing` ilişkisinin
    kullandığı orantısal yaklaşımın diğer kurallara genellenmiş hali.

    Attributes:
        scale_ema: Bbox yüksekliğinin üstel hareketli ortalaması (piksel).
            Henüz hiç güncellenmemişse 0.0; bu durumda kuralların ham bbox
            yüksekliğine (fallback) düşmesi gerekir.
    """

    track_id: int
    class_name: str
    stationary_frames: int = 0
    last_center: tuple[float, float] | None = None
    tip_over_frames: int = 0
    scale_ema: float = 0.0
    flags: Dict[str, Any] = field(default_factory=dict)

    def update(
        self,
        detection: TrackedObject,
        fps: float,
        immobility_window_frames: int = 1,
        movement_ratio_threshold: float = 0.1,
    ) -> None:
        """Track durumunu bir kare için günceller.

        Args:
            detection: Bu karedeki güncel `TrackedObject` (geçmişi de içerir).
            fps: Efektif kare/saniye hızı.
            immobility_window_frames: Hareketsizlik kontrolü için bakılacak
                kümülatif pencere (kare sayısı). Bkz. aşağıdaki not.
            movement_ratio_threshold: Pencere boyunca yer değiştirmenin
                `scale_ema`'ya oranı bu değerin altındaysa "durağan" sayılır.
        """
        det = detection.last_detection

        # Ölçek: bbox yüksekliğinin üstel ortalaması. Her kural kendi
        # başına yeniden hesaplamaz; burada bir kez güncellenir, kurallar
        # (rules.py) sadece okur.
        height = det.height
        if height > 0:
            if self.scale_ema <= 0.0:
                self.scale_ema = height
            else:
                self.scale_ema = SCALE_EMA_ALPHA * height + (1 - SCALE_EMA_ALPHA) * self.scale_ema

        if self.last_center is None:
            self.last_center = det.center
            return

        # Hareketsizlik: anlık (frame-to-frame) fark yerine kümülatif pencere
        # kullanılır. Eski tasarım (`dx < 15 and dy < 15` karşılaştırması, her
        # karede önceki karşılaştırıldığı nokta güncellenir) yavaş ama sürekli
        # hareketi (örn. normal yürüme, ~1-2px/kare) hiç yakalayamıyordu: her
        # kare kendi başına eşiğin altında kaldığı için sayaç asla sıfırlanmıyor,
        # kişi aslında hareket ederken "durağan" sayılıyordu. Kümülatif pencere
        # (`TrackedObject.displacement`) bu küçük adımları toplayarak gerçek
        # net hareketi görünür kılar. Oran (movement_ratio_threshold), sabit
        # piksel eşiği yerine `scale_ema`'ya bölünerek kameraya uzaklıktan
        # bağımsız hale getirilir (aynı `_rule_fall` mantığı).
        if self.scale_ema > 0.0:
            wdx, wdy = detection.displacement(immobility_window_frames)
            movement_ratio = math.hypot(wdx, wdy) / self.scale_ema
            if movement_ratio < movement_ratio_threshold:
                self.stationary_frames += 1
            else:
                self.stationary_frames = 0
        else:
            # Ölçek henüz hesaplanmadıysa (ilk kare) sabit piksel eşiğine düş.
            dx = abs(det.center[0] - self.last_center[0])
            dy = abs(det.center[1] - self.last_center[1])
            if dx < 15 and dy < 15:
                self.stationary_frames += 1
            else:
                self.stationary_frames = 0

        # Devrilme için bbox en/boy oranı takibi
        ar = det.aspect_ratio
        if detection.class_name == "arac":
            if ar > 1.45:
                self.tip_over_frames += 1
            else:
                self.tip_over_frames = max(0, self.tip_over_frames - 1)

        self.last_center = det.center

    def seconds_stationary(self, fps: float) -> float:
        return self.stationary_frames / fps if fps else 0.0


class TrackStateMachine:
    """Tüm track'lerin `TrackState`'ini kareler arasında tutan yönetici.

    `immobility_window_seconds` ve `movement_ratio_threshold`, hareketsizlik
    kontrolünün kümülatif pencere ve ölçeğe oranlı eşiğini belirler (bkz.
    `TrackState.update` docstring'i). `EventEngine`, bu değerleri
    `config.yaml`'daki `events.thresholds.immobility` bloğundan okuyup buraya iletir.
    """

    def __init__(
        self,
        fps: float = 25.0,
        immobility_window_seconds: float = 1.0,
        movement_ratio_threshold: float = 0.1,
    ):
        self.fps = fps
        self.states: Dict[int, TrackState] = {}
        self.immobility_window_frames = max(1, round(immobility_window_seconds * fps))
        self.movement_ratio_threshold = movement_ratio_threshold

    def update(self, tracks: List[TrackedObject]) -> None:
        for t in tracks:
            state = self.states.setdefault(t.track_id, TrackState(t.track_id, t.class_name))
            state.update(t, self.fps, self.immobility_window_frames, self.movement_ratio_threshold)

    def get(self, track_id: int) -> TrackState | None:
        return self.states.get(track_id)
