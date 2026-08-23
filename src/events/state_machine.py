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
        immobility_cfg: Dict[str, Any] | None = None,
        tip_over_cfg: Dict[str, Any] | None = None,
    ) -> None:
        """Track durumunu bir kare için günceller.

        Args:
            detection: Bu karedeki güncel `TrackedObject` (geçmişi de içerir).
            fps: Efektif kare/saniye hızı.
            immobility_cfg: Hareketsizlik kontrolü ayarları
                (`window_frames`, `movement_ratio_threshold`). Bkz. aşağıdaki not.
            tip_over_cfg: Devrilme kontrolü ayarları
                (`window_frames`, `height_drop_ratio`). Bkz. aşağıdaki not.

        Not: Her kural kendi ayarlarını isimli bir alt sözlükten okur (aynı
        `src/events/rules.py` içindeki `RuleSet`'in `self.thresholds.get("fall", {})`
        deseni). Bu, yeni bir kural eklendiğinde yalnızca ilgili `cfg` sözlüğüne
        bir anahtar eklemeyi gerektirir; imzaya sırayla yeni pozisyonel parametre
        eklemek gerekmez — sıra karışması riski (yanlış değerin yanlış kurala
        gitmesi) böylece ortadan kalkar.
        """
        immobility_cfg = immobility_cfg or {}
        tip_over_cfg = tip_over_cfg or {}
        immobility_window_frames = immobility_cfg.get("window_frames", 1)
        movement_ratio_threshold = immobility_cfg.get("movement_ratio_threshold", 0.1)
        tip_over_window_frames = tip_over_cfg.get("window_frames", 1)
        tip_over_height_drop_ratio = tip_over_cfg.get("height_drop_ratio", 0.3)

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

        # Devrilme: en/boy oranının GENİŞLEMESİ tek başına yeterli kanıt değildir.
        # Forklift kameraya doğru dönerken de (yaw) önden görünümden (dar, ar~0.4)
        # yandan görünüme (geniş, ar~1.3) geçer — bu, gerçek devrilmeyle (roll)
        # aynı görüntüsel etkiyi (genişleme) yaratır ama forklift fiziksel olarak
        # yatmamıştır (bkz. proximity.mp4 gözlemi: aspect_ratio 0.37'den 1.31'e
        # ~40 karede düzgün bir eğriyle çıkıyor, forklift sadece yön değiştiriyor).
        #
        # Ayırt edici gerçek fiziksel kanıt: devrilirken forkliftin üst kısmı
        # (kabin, direksiyon) ANİDEN yere yaklaşır, bbox yüksekliği pencere
        # boyunca hızla düşer. Dönüşte yükseklik nispeten stabil kalır (aynı
        # gözlemde h: 369->328->~317px, yalnızca ~%14 yavaş bir daralma).
        # Bu yüzden iki koşul BİRLİKTE aranır: oran artışı VE eşzamanlı hızlı
        # yükseklik düşüşü. Yalnızca oran artışı sayaç yükseltmez.
        ar = det.aspect_ratio
        if detection.class_name == "arac":
            past_det = detection.detection_at_offset(tip_over_window_frames)
            past_height = past_det.height
            height_drop_ratio = (past_height - det.height) / past_height if past_height > 0 else 0.0
            is_widening = ar > 1.45
            is_collapsing = height_drop_ratio >= tip_over_height_drop_ratio

            if is_widening and is_collapsing:
                self.tip_over_frames += 1
            else:
                self.tip_over_frames = max(0, self.tip_over_frames - 1)

        self.last_center = det.center

    def seconds_stationary(self, fps: float) -> float:
        return self.stationary_frames / fps if fps else 0.0


class TrackStateMachine:
    """Tüm track'lerin `TrackState`'ini kareler arasında tutan yönetici.

    `thresholds`, `config.yaml`'daki `events.thresholds` bloğunun tamamıdır
    (aynı sözlük `src/events/rules.py` içindeki `RuleSet`'e de geçirilir).
    Her kural kendi alt bloğunu (`thresholds["immobility"]`, `thresholds["tip_over"]`
    gibi) isimle okur; yeni bir kural saniye cinsinden bir pencere parametresi
    gerektirdiğinde, bu sınıfa veya `TrackState.update()`'in imzasına yeni bir
    pozisyonel parametre eklemek GEREKMEZ — sadece ilgili `_frame_window()`
    çağrısı ve `TrackState.update()` içinde `cfg.get(...)` okuması eklenir.
    """

    def __init__(self, fps: float = 25.0, thresholds: Dict[str, Any] | None = None):
        self.fps = fps
        self.thresholds = thresholds or {}
        self.states: Dict[int, TrackState] = {}

    def _frame_window(self, rule_name: str, key: str = "window_seconds", default: float = 1.0) -> int:
        """Bir kuralın saniye cinsinden pencere ayarını kare sayısına çevirir."""
        seconds = self.thresholds.get(rule_name, {}).get(key, default)
        return max(1, round(seconds * self.fps))

    def update(self, tracks: List[TrackedObject]) -> None:
        immobility_cfg = self.thresholds.get("immobility", {})
        tip_over_cfg = self.thresholds.get("tip_over", {})

        immobility_update_cfg = {
            "window_frames": self._frame_window("immobility", default=1.0),
            "movement_ratio_threshold": immobility_cfg.get("movement_ratio_threshold", 0.1),
        }
        tip_over_update_cfg = {
            "window_frames": self._frame_window("tip_over", default=0.5),
            "height_drop_ratio": tip_over_cfg.get("height_drop_ratio", 0.3),
        }

        for t in tracks:
            state = self.states.setdefault(t.track_id, TrackState(t.track_id, t.class_name))
            state.update(t, self.fps, immobility_update_cfg, tip_over_update_cfg)

    def get(self, track_id: int) -> TrackState | None:
        return self.states.get(track_id)
