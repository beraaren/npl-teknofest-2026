"""Nesne takip wrapper'ı (ByteTrack/BoT-SORT)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .detector import Detection


class TrackedObject:
    """Bir track ID'ye ait zaman içindeki durum."""

    def __init__(self, track_id: int, class_name: str, initial_detection: Detection):
        self.track_id = track_id
        self.class_name = class_name
        self.history: List[Detection] = [initial_detection]
        self.disappeared = 0

    def update(self, detection: Detection) -> None:
        self.history.append(detection)
        self.disappeared = 0

    @property
    def last_detection(self) -> Detection:
        return self.history[-1]

    @property
    def center_history(self) -> List[tuple[float, float]]:
        return [d.center for d in self.history]

    @property
    def speed(self) -> tuple[float, float]:
        """Son iki tespit arasındaki piksel farkı."""
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

    @property
    def is_stationary(self, threshold_pixels: float = 15.0, window: int = 5) -> bool:
        recent = self.history[-window:]
        if len(recent) < 2:
            return False
        first = recent[0].center
        return all(
            abs(d.center[0] - first[0]) < threshold_pixels
            and abs(d.center[1] - first[1]) < threshold_pixels
            for d in recent[1:]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class": self.class_name,
            "history_length": len(self.history),
            "last_center": [round(self.last_detection.center[0], 2), round(self.last_detection.center[1], 2)],
            "speed": [round(self.speed[0], 2), round(self.speed[1], 2)],
        }


class ObjectTracker:
    """Ultralytics tracker entegrasyonu."""

    def __init__(self, tracker_name: str = "bytetrack", persist: bool = True):
        self.tracker_name = tracker_name
        self.persist = persist
        self._model = None

    def _get_tracker_str(self) -> str:
        # Ultralytics yerleşik tracker isimleri yaml uzantısı ister ("bytetrack.yaml")
        name = self.tracker_name
        if not name.endswith((".yaml", ".yml")):
            name = f"{name}.yaml"
        return name

    def track(self, frame: NDArray[np.uint8], detector: Any, frame_idx: int = 0) -> List[TrackedObject]:
        """Tespit + takip tek adımda yapılır."""
        model = detector._load()
        results = model.track(
            frame,
            verbose=False,
            conf=detector.confidence,
            tracker=self._get_tracker_str(),
            persist=self.persist,
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
