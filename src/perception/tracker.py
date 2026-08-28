"""Nesne takip wrapper'ı (ByteTrack/BoT-SORT) ve track durum modeli.

Bu modül, Kanal A algı hattının takip adımını oluşturur:
:class:`ObjectTracker` her karede tespit + takip kimliği ataması yapar
(Ultralytics'in yerleşik ``model.track()`` çağrısı üzerinden), sonuçlar
:class:`TrackedObject` içinde birikimli bir geçmiş (``history``) olarak
saklanır.

``TrackedObject`` sadece takip katmanının değil, **olay motorunun**
(`src/events/`) da temel veri kaynağıdır. Kural motorundaki tüm kinematik
hesaplar (düşme, devrilme) bu sınıfın :attr:`history`
listesine ve pencereli erişim metodlarına (:meth:`TrackedObject.displacement`,
:meth:`TrackedObject.detection_at_offset`) dayanır — bu yüzden bu sınıfın
geçmişinin kareler arasında **doğru şekilde korunması** kritik önemdedir. Bu
geçmiş yanlışlıkla sıfırlanırsa (örn. her karede yeni bir ``TrackedObject``
örneği yaratılırsa), pencereli hesapların tümü tek elemanlı geçmişe düşer ve
hıza/yer değiştirmeye dayalı tüm kurallar sessizce hiç tetiklenemez hale gelir
(bkz. `src/events/event_engine.py` — `_tracked_objects` kalıcı sözlüğü tam
olarak bu riski önlemek için tasarlanmıştır).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .detector import Detection


class TrackedObject:
    """Bir takip kimliğine (track ID) ait zaman içindeki birikimli durum.

    Bir nesnenin video boyunca gözlenen tüm :class:`Detection` kayıtlarını
    :attr:`history` listesinde sırayla tutar. Bu liste, olay motorunun
    kinematik kurallarının (hız, yer değiştirme, en/boy oranı değişimi)
    veri kaynağıdır.

    Attributes:
        track_id: Bu nesneye atanmış, video boyunca sabit kalan takip kimliği.
        class_name: Nesnenin kanonik sınıf adı (örn. ``"insan"``, ``"arac"``).
            Araç isimlendirme adımı (`vehicle_labeler.py`) bu alanı ve
            geçmişteki ilgili kayıtları sonradan güncelleyebilir.
        history: Bu track için sırayla biriken :class:`Detection` kayıtları.
            İlk eleman :meth:`__init__`'e verilen başlangıç algılamasıdır.
        disappeared: Nesnenin son kaç kare üst üste **görülmediğini** sayan
            sayaç. :class:`src.perception.observer_agent.ObserverAgent`
            bu sayaç 5'e ulaştığında nesneyi aktif listeden çıkarır; kısa
            süreli görünmezlik (occlusion) toleransı sağlar.
    """

    def __init__(self, track_id: int, class_name: str, initial_detection: Detection):
        """Yeni bir track kaydı oluşturur.

        Args:
            track_id: Takip kimliği.
            class_name: Kanonik sınıf adı.
            initial_detection: Bu track'in ilk gözlemlendiği karedeki algılama;
                :attr:`history`'nin ilk elemanı olur.
        """
        self.track_id = track_id
        self.class_name = class_name
        self.history: List[Detection] = [initial_detection]
        self.disappeared = 0

    def update(self, detection: Detection) -> None:
        """Yeni bir karedeki algılamayı geçmişe ekler ve kayboluş sayacını sıfırlar.

        Bu metod **geçmişi asla sıfırlamaz**, sadece ekler — pencereli
        kinematik hesapların (bkz. modül docstring'i) doğru çalışması bu
        davranışa bağlıdır.

        Args:
            detection: Bu karedeki güncel algılama.
        """
        self.history.append(detection)
        self.disappeared = 0

    @property
    def last_detection(self) -> Detection:
        """Geçmişteki en güncel (son) algılama kaydı."""
        return self.history[-1]

    @property
    def center_history(self) -> List[tuple[float, float]]:
        """Geçmişteki tüm algılamaların merkez koordinatları, sırayla."""
        return [d.center for d in self.history]

    @property
    def speed(self) -> tuple[float, float]:
        """Son iki ardışık algılama arasındaki anlık merkez farkı (piksel/kare).

        Bu, **tek karelik** bir hız ölçümüdür. Birkaç kareye yayılan
        hareketleri (örn. düşme) yakalamak için yeterli değildir; bu amaçla
        :meth:`displacement` (kümülatif pencere) kullanılmalıdır.

        Returns:
            ``(dx, dy)``: son iki karenin merkezleri arasındaki fark. Geçmişte
            tek kayıt varsa ``(0.0, 0.0)``.
        """
        if len(self.history) < 2:
            return (0.0, 0.0)
        c1 = self.history[-2].center
        c2 = self.history[-1].center
        return (c2[0] - c1[0], c2[1] - c1[1])

    def detection_at_offset(self, window_frames: int) -> Detection:
        """`window_frames` kare öncesindeki (veya en eski mevcut) `Detection`'ı döner.

        `displacement()` ile aynı pencere indeksleme mantığını kullanır, ama
        sadece merkez farkını değil, tüm `Detection` nesnesini (bbox, aspect_ratio,
        height dahil) döndürür. Bu, bir pencere boyunca birden fazla geometrik
        özelliğin (örn. en/boy oranı VE yükseklik) birlikte karşılaştırılması
        gerektiğinde tekrar tekrar indeks hesaplamaktan kaçınmak için kullanılır
        (bkz. `TrackState.update` — forklift devrilme kontrolü).

        Args:
            window_frames: Kaç kare öncesine bakılacağı (>= 1).

        Returns:
            İlgili karedeki `Detection`. Geçmiş `window_frames`'den kısaysa en
            eski kayıt döner.
        """
        idx = max(0, len(self.history) - 1 - max(1, window_frames))
        return self.history[idx]

    def displacement(self, window_frames: int) -> tuple[float, float]:
        """`window_frames` kare öncesine göre kümülatif yer değiştirme.

        Tek karelik `speed`'in aksine, birkaç kareye yayılan hareketleri
        (örn. düşme) yakalamak için kısa bir pencere boyunca birikimli
        farkı döner. Geçmiş `window_frames`'den kısaysa mevcut en eski
        kayıt referans alınır — video başında sahte sıfır yer değiştirme
        yerine gerçek kısmi hareket yansıtılır.

        Args:
            window_frames: Kaç kare öncesine bakılacağı (>= 1).

        Returns:
            `(dx, dy)`: pencere başı ile şu anki merkez arasındaki fark.
            Geçmişte tek kayıt varsa `(0.0, 0.0)`.
        """
        if len(self.history) < 2:
            return (0.0, 0.0)
        idx = max(0, len(self.history) - 1 - max(1, window_frames))
        c1 = self.history[idx].center
        c2 = self.history[-1].center
        return (c2[0] - c1[0], c2[1] - c1[1])

    def to_dict(self) -> dict[str, Any]:
        """Track durumunu gözlem (observation) çıktısı için özetler.

        Bu sözlük, :class:`src.perception.observer_agent.ObserverAgent`
        tarafından üretilen kare başına gözlem dict'inin ``tracks`` alanına
        girer ve olay motoruna, VLM prompt'una, nihai rapora aktarılır.

        Returns:
            ``track_id``, ``class``, ``history_length``, ``last_center``,
            ``speed`` anahtarlarını içeren özet sözlük. Tüm sayısal değerler
            raporlama için yuvarlanır.
        """
        return {
            "track_id": self.track_id,
            "class": self.class_name,
            "history_length": len(self.history),
            "last_center": [round(self.last_detection.center[0], 2), round(self.last_detection.center[1], 2)],
            "speed": [round(self.speed[0], 2), round(self.speed[1], 2)],
        }


class ObjectTracker:
    """Ultralytics'in yerleşik takip (ByteTrack/BoT-SORT) entegrasyonunu sarmalayan wrapper.

    Bu sınıf, :class:`src.perception.detector.ObjectDetector` (Ultralytics
    backend'i) ile birlikte kullanılmak üzere tasarlanmıştır; tespit ve takip
    tek bir Ultralytics ``model.track()`` çağrısında birleştirilir. HuggingFace
    gibi ``supports_tracking=False`` olan backend'lerde bu sınıf
    **kullanılmaz**; onun yerine
    :meth:`src.perception.observer_agent.ObserverAgent._track_by_iou` devreye
    girer.

    Attributes:
        tracker_name: Ultralytics'in yerleşik tracker adı (``"bytetrack"`` veya
            ``"botsort"``). Dosya uzantısı otomatik eklenir (bkz. :meth:`_get_tracker_str`).
        persist: ``True`` ise, aynı model örneği üzerinde ardışık çağrılarda
            takip durumu (track ID atamaları) korunur — video karelerini
            sırayla işlerken bu ``True`` olmalıdır, aksi halde her karede
            takip sıfırdan başlar ve kimlikler tutarsız hale gelir.
    """

    def __init__(self, tracker_name: str = "bytetrack", persist: bool = True):
        """Tracker'ı yapılandırır.

        Args:
            tracker_name: ``"bytetrack"`` veya ``"botsort"``.
            persist: Ardışık çağrılar arasında takip durumunun korunup
                korunmayacağı.
        """
        self.tracker_name = tracker_name
        self.persist = persist
        self._model = None

    def _get_tracker_str(self) -> str:
        """Tracker adını Ultralytics'in beklediği ``.yaml`` uzantılı biçime çevirir.

        Returns:
            Örn. ``"bytetrack"`` verilirse ``"bytetrack.yaml"``.
        """
        # Ultralytics yerleşik tracker isimleri yaml uzantısı ister ("bytetrack.yaml")
        name = self.tracker_name
        if not name.endswith((".yaml", ".yml")):
            name = f"{name}.yaml"
        return name

    def track(self, frame: NDArray[np.uint8], detector: Any, frame_idx: int = 0) -> List[TrackedObject]:
        """Tek bir karede tespit ve takip kimliği atamasını birlikte yapar.

        Bu metod, verilen ``detector``'ın (Ultralytics tabanlı bir
        :class:`src.perception.detector.ObjectDetector` olmalıdır) dahili
        modelini ve sınıf-eşleme mantığını (``detector._map_class``)
        doğrudan kullanır; bu yüzden ``detector.supports_tracking`` ``True``
        olmalıdır.

        Not: Bu metodun döndürdüğü her :class:`TrackedObject`, tek elemanlı
        yeni bir geçmişle (sadece bu karenin algılamasıyla) oluşturulur.
        Kareler arasında geçmişin **birikmesi**,
        :class:`src.perception.observer_agent.ObserverAgent.observe_frame`
        içindeki birleştirme (merge) adımının sorumluluğundadır — bu metod
        tek başına çağrıldığında track geçmişi kalıcı olmaz.

        Args:
            frame: RGB, ``HWC`` düzeninde ``uint8`` kare.
            detector: ``supports_tracking=True`` olan bir tespit backend'i
                (model erişimi için ``_load()``, sınıf eşleme için
                ``_map_class()`` ve eşik için ``confidence`` özelliğini
                kullanır).
            frame_idx: Bu karenin indeksi.

        Returns:
            Karede bulunan her takip edilen nesne için, tek elemanlı bir
            geçmişle başlatılmış :class:`TrackedObject`. Hiç takip
            kimliği atanamadıysa (``results.boxes.id is None``) boş liste
            döner.
        """
        model = detector._load()
        import torch
        device = getattr(detector, "device", 0 if torch.cuda.is_available() else "cpu")
        results = model.track(
            frame,
            verbose=False,
            conf=detector.confidence,
            tracker=self._get_tracker_str(),
            persist=self.persist,
            device=device,
        )[0]

        tracked: List[TrackedObject] = []
        if results.boxes is None or results.boxes.id is None:
            return tracked

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        ids = results.boxes.id.cpu().numpy().astype(int)
        names = results.names

        for box, conf, cls_idx, tid in zip(boxes, confs, classes, ids):
            class_name = names.get(cls_idx, str(cls_idx))
            class_name = detector._map_class(class_name)
            det = Detection(
                class_name=class_name,
                confidence=float(conf),
                bbox=tuple(float(v) for v in box),
                frame_idx=frame_idx,
                track_id=int(tid),
            )
            tracked.append(TrackedObject(track_id=int(tid), class_name=class_name, initial_detection=det))

        return tracked
