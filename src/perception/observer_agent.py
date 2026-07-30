"""Gözlemci Ajan: videoyu izler, tespit/takip/scene graph üretir."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from numpy.typing import NDArray

from ..config import PerceptionConfig
from ..utils.logger import get_logger
from .detector import Detection, create_detector
from .scene_graph import SceneGraph
from .tracker import ObjectTracker, TrackedObject


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """İki bbox (x1,y1,x2,y2) arası IoU."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class ObserverAgent:
    """Objektif algı katmanı; sadece gözlemlenen nesneleri ve ilişkileri raporlar."""

    def __init__(self, config: PerceptionConfig):
        self.config = config
        self.logger = get_logger("ObserverAgent")
        self.detector = create_detector(config)
        self.tracker = ObjectTracker(tracker_name=config.tracker, persist=config.tracker_persist)
        self.tracks: Dict[int, TrackedObject] = {}
        self._next_track_id = 0

    def observe_frame(self, frame: NDArray[np.uint8], frame_idx: int, timestamp: float) -> Dict[str, Any]:
        """Tek kare için tespit, takip ve scene graph üretir."""
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

        # Scene graph
        detections: List[Detection] = [t.last_detection for t in self.tracks.values() if t.disappeared < 5]
        graph = SceneGraph.from_detections(frame_idx, timestamp, detections)

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
        """Kareleri işler. sampled_indices verilirse timestamp gerçek video
        kare indeksinden hesaplanır (örneklem indeksi değil)."""
        observations = []
        for idx, frame in enumerate(frames):
            real_idx = sampled_indices[idx] if sampled_indices else idx
            timestamp = real_idx / fps if fps else 0.0
            obs = self.observe_frame(frame, idx, timestamp)
            observations.append(obs)
        return observations

    def _track_by_iou(self, detections: List[Detection], min_iou: float = 0.3) -> List[TrackedObject]:
        """Aynı sınıftaki aktif track'lere IoU ile eşle; yoksa yeni ID ver.

        Ultralytics yolundaki gibi her karede yeni TrackedObject döner;
        geçmiş birleştirmesi observe_frame'deki merge döngüsünde yapılır.
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
            tracked.append(TrackedObject(track_id=best_id, class_name=det.class_name, initial_detection=det))
        return tracked
