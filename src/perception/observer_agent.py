"""Gözlemci Ajan: videoyu izler, tespit/takip/sahne grafiği üretir.

Bu modül, Kanal A algı hattının orkestratörüdür:
:class:`src.perception.detector.ObjectDetector` (veya
:class:`src.perception.hf_detector.HFObjectDetector`) ile tespit,
:class:`src.perception.tracker.ObjectTracker` ile takip ve
:class:`src.perception.scene_graph.SceneGraph` ile ilişki çıkarımını tek bir
kare-başına-gözlem (observation) sözlüğünde birleştirir.

``ObserverAgent``'ın ürettiği observation dict'i, kod tabanındaki en önemli
arayüz sözleşmelerinden biridir: `src/events/event_engine.py`
(:class:`~src.events.event_engine.EventEngine`), VLM prompt'u ve nihai analiz
raporu **hepsi** bu sözlüğün biçimine (``frame_idx``, ``timestamp``,
``detections``, ``tracks``, ``scene_graph``) bağımlıdır. Bu alanlardan
birini kaldırmak veya yeniden adlandırmak, aşağı akıştaki tüketicilerin
tümünü bozar.

``ObserverAgent`` **objektif** bir katmandır: hiçbir risk yorumu, karar veya
eşik kontrolü yapmaz — sadece gözlemlenebilir olguları (hangi nesne nerede,
hangi nesne hangi nesneyle ilişkili) raporlar. Yorumlama ve karar verme
`src/events/` ve `src/reasoning/` katmanlarının sorumluluğudur.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from numpy.typing import NDArray

from ..config import PerceptionConfig
from ..utils.logger import get_logger
from .detector import Detection, create_detector
from .scene_graph import DEFAULT_PROXIMITY_THRESHOLD, SceneGraph
from .tracker import ObjectTracker, TrackedObject


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """İki sınırlayıcı kutu (``x1, y1, x2, y2``) arasındaki Intersection-over-Union.

    :meth:`ObserverAgent._track_by_iou` içinde, takip desteği olmayan
    backend'lerde (örn. HF transformers) aynı nesnenin ardışık karelerde
    tanınması için kullanılır.

    Args:
        a: Birinci kutu, piksel koordinatlarında.
        b: İkinci kutu, piksel koordinatlarında.

    Returns:
        0.0-1.0 aralığında IoU değeri. Kutulardan biri dejenere (alanı
        sıfır) veya kesişim yoksa ``0.0``.
    """
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class ObserverAgent:
    """Objektif algı katmanı; sadece gözlemlenen nesneleri ve ilişkileri raporlar.

    Bu sınıf, video karelerini sırayla işleyip her kare için bir gözlem
    (observation) sözlüğü üretir. İçeride şu durumu kalıcı olarak tutar:

      * :attr:`detector` — tespit backend'i (bir kez oluşturulur, tüm video
        boyunca yeniden kullanılır).
      * :attr:`tracker` — Ultralytics takip wrapper'ı (backend takip
        desteklediğinde kullanılır).
      * :attr:`tracks` — track ID'sine göre indekslenmiş, video boyunca
        kalıcı :class:`~src.perception.tracker.TrackedObject` sözlüğü. Bu
        sözlük, `src/events/event_engine.py` içindeki ``_tracked_objects``
        deseninin algı katmanındaki karşılığıdır: her karede sıfırdan
        yaratılmaz, sadece güncellenir — aksi halde track geçmişi (ve ona
        bağlı tüm kinematik hesaplar) kaybolur.

    Attributes:
        config: Algı katmanı yapılandırması.
        proximity_threshold: Gözlem çıktısındaki sahne grafiğinin ``near``
            kenarları için kullandığı merkez mesafesi eşiği (piksel).
        detector: :func:`~src.perception.detector.create_detector` ile
            oluşturulan tespit backend'i örneği.
        tracker: :class:`~src.perception.tracker.ObjectTracker` örneği.
        tracks: ``{track_id: TrackedObject}`` biçiminde kalıcı takip durumu.
    """

    def __init__(
        self,
        config: PerceptionConfig,
        proximity_threshold: float = DEFAULT_PROXIMITY_THRESHOLD,
    ):
        """Gözlemci ajanı başlatır.

        Args:
            config: Algı katmanı yapılandırması (detector backend, tracker vb.).
            proximity_threshold: Sahne grafiği `near` kenarları için merkez mesafesi
                eşiği (piksel). Olay motoru grafiği kendi yapılandırma eşiğiyle
                yeniden kurduğu için buradaki değer yalnızca gözlem JSON'una ve VLM
                prompt'una yansır; ikisinin aynı olması raporlarda tutarlılık sağlar.
        """
        self.config = config
        self.proximity_threshold = proximity_threshold
        self.logger = get_logger("ObserverAgent")
        self.detector = create_detector(config)
        self.tracker = ObjectTracker(tracker_name=config.tracker, persist=config.tracker_persist)
        self.tracks: Dict[int, TrackedObject] = {}
        self._next_track_id = 0

    def observe_frame(self, frame: NDArray[np.uint8], frame_idx: int, timestamp: float) -> Dict[str, Any]:
        """Tek bir kare için tespit, takip ve sahne grafiği üretir.

        İşlem adımları:

          1. Backend takip destekliyorsa (:attr:`detector.supports_tracking`),
             :attr:`tracker` ile tespit+takip birlikte yapılır. Desteklemiyorsa
             (örn. HF transformers), önce ham tespit yapılır ve
             :meth:`_track_by_iou` ile geçmiş track'lere IoU tabanlı eşleme
             uygulanır.
          2. Bu karede görülen her track, kalıcı :attr:`tracks` sözlüğünde
             güncellenir (``update``) veya ilk kez görülüyorsa eklenir. Bu
             adım, track geçmişinin (ve ona bağlı tüm kinematik hesapların)
             kareler arasında korunmasını sağlar.
          3. Bu karede görülmeyen track'lerin :attr:`~TrackedObject.disappeared`
             sayacı artırılır; 5 kareyi aşan track'ler aktif listelerden
             (döndürülen ``detections``/``tracks``) düşürülür — kısa süreli
             görünmezliğe (occlusion) tolerans sağlanır.
          4. Aktif track'lerin son algılamalarından bir
             :class:`~src.perception.scene_graph.SceneGraph` kurulur.

        Args:
            frame: RGB, ``HWC`` düzeninde ``uint8`` kare.
            frame_idx: Bu karenin indeksi.
            timestamp: Karenin video başlangıcına göre saniye cinsinden zamanı.

        Returns:
            ``frame_idx``, ``timestamp``, ``detections`` (aktif
            :class:`~src.perception.detector.Detection` sözlükleri),
            ``tracks`` (aktif :class:`~src.perception.tracker.TrackedObject`
            özetleri) ve ``scene_graph`` (:meth:`SceneGraph.to_dict` çıktısı)
            anahtarlarını içeren gözlem sözlüğü. Bu biçim, olay motorunun ve
            VLM prompt'unun beklediği sabit sözleşmedir.
        """
        # Tespit (ve takip)
        if getattr(self.detector, "supports_tracking", False):
            tracked = self.tracker.track(frame, self.detector, frame_idx=frame_idx)
        else:
            # Ultralytics dışı backend'lerde (örn. HF transformers) model.track()
            # yoktur; basit IoU eşleşmesiyle track ID'leri korunur.
            tracked = self._track_by_iou(self.detector.detect(frame, frame_idx=frame_idx))

        # Track state güncelle
        for t in tracked:
            existing = self.tracks.get(t.track_id)
            if existing:
                existing.update(t.last_detection)
            else:
                self.tracks[t.track_id] = t

        # Kaybolan track'leri işaretle
        active_ids = {t.track_id for t in tracked}
        for tid, t in self.tracks.items():
            if tid not in active_ids:
                t.disappeared += 1
            else:
                t.disappeared = 0

        # Scene graph
        detections: List[Detection] = [t.last_detection for t in self.tracks.values() if t.disappeared < 5]
        graph = SceneGraph.from_detections(
            frame_idx, timestamp, detections, proximity_threshold=self.proximity_threshold
        )

        return {
            "frame_idx": frame_idx,
            "timestamp": round(timestamp, 2),
            "detections": [d.to_dict() for d in detections],
            "tracks": [t.to_dict() for t in self.tracks.values() if t.disappeared < 5],
            "scene_graph": graph.to_dict(),
        }

    def observe_video(
        self,
        frames: List[NDArray[np.uint8]],
        fps: float,
        sampled_indices: List[int] | None = None,
    ) -> List[Dict[str, Any]]:
        """Bir kare listesini sırayla işleyip gözlem listesi üretir.

        Önemli: ``sampled_indices`` verildiğinde, her gözlemin ``timestamp``
        değeri **gerçek video kare indeksinden** (``sampled_indices[idx]``)
        hesaplanır — listedeki sıra pozisyonundan (``idx``) değil. Bu ayrım
        kritiktir: çağıran kod genellikle bir örnekleme/atlama adımıyla
        (örn. her 2. kareyi alarak) seçilmiş bir alt küme kareyi buraya
        verir; ``timestamp``'in gerçek videodaki zamana karşılık gelmesi
        için gerçek kare indeksi kullanılmalıdır (bkz. `benioku.md` §5.1).

        Args:
            frames: İşlenecek RGB kare listesi (sıralı).
            fps: Videonun kare/saniye hızı (zaman damgası hesabı için).
            sampled_indices: ``frames`` listesindeki her elemanın gerçek
                video kare indeksi. ``None`` verilirse liste pozisyonu
                (``idx``) kare indeksi olarak kabul edilir.

        Returns:
            Her kare için :meth:`observe_frame` çıktısı, giriş sırasıyla.
        """
        observations = []
        for idx, frame in enumerate(frames):
            real_idx = sampled_indices[idx] if sampled_indices else idx
            timestamp = real_idx / fps if fps else 0.0
            obs = self.observe_frame(frame, idx, timestamp)
            observations.append(obs)
        return observations

    def _track_by_iou(self, detections: List[Detection], min_iou: float = 0.3) -> List[TrackedObject]:
        """Takip desteği olmayan backend'ler için basit IoU tabanlı takip kimliği ataması.

        Her yeni algılama, aynı sınıftan ve henüz bu karede kullanılmamış,
        henüz kaybolmamış (``disappeared < 5``) aktif track'lerle karşılaştırılır;
        en yüksek IoU'ya sahip olan (``min_iou`` eşiğini geçiyorsa) eşleşme
        kabul edilir. Eşleşme bulunamazsa yeni bir track ID atanır.

        Not: Ultralytics'in ``track()`` yolundaki gibi, bu metot her çağrıda
        **yeni** :class:`TrackedObject` örnekleri döndürür (tek elemanlı
        geçmişle). Geçmişin kareler arasında birikmesi, çağıran
        :meth:`observe_frame` içindeki birleştirme (merge) adımının
        sorumluluğundadır.

        Args:
            detections: Bu karede yapılan ham algılamalar (henüz takip
                kimliği atanmamış).
            min_iou: Bir eşleşmenin kabul edilmesi için gereken minimum IoU.

        Returns:
            Her algılama için, takip kimliği atanmış (mevcut veya yeni)
            :class:`TrackedObject` listesi.
        """
        tracked: List[TrackedObject] = []
        used_ids: set[int] = set()
        for det in detections:
            best_id, best_iou = None, min_iou
            for tid, t in self.tracks.items():
                if tid in used_ids or t.disappeared >= 5 or t.class_name != det.class_name:
                    continue
                iou = _iou(t.last_detection.bbox, det.bbox)
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_id is None:
                best_id = self._next_track_id
                self._next_track_id += 1
            used_ids.add(best_id)
            det.track_id = best_id
            tracked.append(TrackedObject(track_id=best_id, class_name=det.class_name, initial_detection=det))
        return tracked
