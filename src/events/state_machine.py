"""Track bazlı durum makinesi."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..perception.tracker import TrackedObject


@dataclass
class TrackState:
    track_id: int
    class_name: str
    stationary_frames: int = 0
    last_center: tuple[float, float] | None = None
    tip_over_frames: int = 0
    fall_frames: int = 0
    flags: Dict[str, Any] = field(default_factory=dict)

    def update(self, detection: TrackedObject, fps: float) -> None:
        det = detection.last_detection
        if self.last_center is None:
            self.last_center = det.center
            return

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

        # Düşme: dikeyde hızlı düşüş
        if detection.class_name == "insan":
            speed_y = detection.speed[1]
            if speed_y > 30:
                self.fall_frames += 1
            else:
                self.fall_frames = max(0, self.fall_frames - 1)

        self.last_center = det.center

    def seconds_stationary(self, fps: float) -> float:
        return self.stationary_frames / fps if fps else 0.0


class TrackStateMachine:
    def __init__(self, fps: float = 25.0):
        self.fps = fps
        self.states: Dict[int, TrackState] = {}

    def update(self, tracks: List[TrackedObject]) -> None:
        for t in tracks:
            state = self.states.setdefault(t.track_id, TrackState(t.track_id, t.class_name))
            state.update(t, self.fps)

    def get(self, track_id: int) -> TrackState | None:
        return self.states.get(track_id)
