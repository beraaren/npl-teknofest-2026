"""Algı katmanı testleri: detector factory, timestamp, IoU takip."""
import numpy as np

from src.config import PerceptionConfig
from src.perception.detector import ObjectDetector, create_detector
from src.perception.hf_detector import HFObjectDetector
from src.perception.observer_agent import ObserverAgent, _iou


def test_factory_returns_ultralytics_by_default():
    cfg = PerceptionConfig(detector_backend="ultralytics")
    detector = create_detector(cfg)
    assert isinstance(detector, ObjectDetector)
    assert detector.supports_tracking is True


def test_factory_returns_hf_detector():
    cfg = PerceptionConfig(detector_backend="hf_transformers")
    detector = create_detector(cfg)
    assert isinstance(detector, HFObjectDetector)
    assert detector.supports_tracking is False
    assert detector.model_path == cfg.hf_model


def test_observe_video_uses_real_timestamps(monkeypatch):
    """sampled_indices verildiğinde timestamp gerçek video karesinden hesaplanmalı."""
    cfg = PerceptionConfig(detector_backend="hf_transformers")
    observer = ObserverAgent(cfg)

    monkeypatch.setattr(
        observer,
        "observe_frame",
        lambda frame, frame_idx, timestamp: {"frame_idx": frame_idx, "timestamp": timestamp},
    )

    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    sampled_indices = [0, 25, 50]
    observations = observer.observe_video(frames, fps=25.0, sampled_indices=sampled_indices)

    timestamps = [obs["timestamp"] for obs in observations]
    assert timestamps == [0.0, 1.0, 2.0]  # sampled_indices[idx] / fps


def test_observe_video_without_indices_falls_back_to_position(monkeypatch):
    cfg = PerceptionConfig(detector_backend="hf_transformers")
    observer = ObserverAgent(cfg)
    monkeypatch.setattr(
        observer,
        "observe_frame",
        lambda frame, frame_idx, timestamp: {"frame_idx": frame_idx, "timestamp": timestamp},
    )
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
    observations = observer.observe_video(frames, fps=10.0)
    assert [obs["timestamp"] for obs in observations] == [0.0, 0.1]


def test_iou_tracker_keeps_ids_across_frames():
    from src.perception.detector import Detection

    cfg = PerceptionConfig(detector_backend="hf_transformers")
    observer = ObserverAgent(cfg)

    det1 = Detection(class_name="insan", confidence=0.9, bbox=(10, 10, 50, 50))
    det2 = Detection(class_name="insan", confidence=0.9, bbox=(12, 12, 52, 52))  # örtüşüyor
    det3 = Detection(class_name="insan", confidence=0.9, bbox=(200, 200, 240, 240))  # uzak

    tracked1 = observer._track_by_iou([det1])
    assert tracked1[0].track_id == 0

    # İlk karedeki merge'i taklit et
    observer.tracks[0] = tracked1[0]

    tracked2 = observer._track_by_iou([det2, det3])
    ids = {t.track_id for t in tracked2}
    assert 0 in ids  # örtüşen tespit aynı ID'yi korur
    assert len(ids) == 2  # uzak tespit yeni ID alır


def test_iou_helper():
    a = (0, 0, 10, 10)
    b = (5, 5, 15, 15)
    assert 0 < _iou(a, b) < 1
    assert _iou(a, a) == 1.0
    assert _iou(a, (100, 100, 110, 110)) == 0.0
