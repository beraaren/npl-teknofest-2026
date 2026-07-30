"""Kritik kare seçimi testleri."""
import numpy as np

from src.preprocessing.critical_frames import _timestamp_to_seconds, select_critical_frames


def _frames(n: int, size: int = 64) -> list:
    return [np.random.randint(0, 255, (size, size, 3), dtype=np.uint8) for _ in range(n)]


def test_timestamp_parser():
    assert _timestamp_to_seconds("01:30") == 90.0
    assert _timestamp_to_seconds("00:05") == 5.0
    assert _timestamp_to_seconds("01:00:00") == 3600.0
    assert _timestamp_to_seconds(42) == 42.0
    assert _timestamp_to_seconds("gecersiz") is None
    assert _timestamp_to_seconds(None) is None


def test_event_frames_are_prioritized():
    frames = _frames(8)
    sampled_indices = [0, 10, 20, 30, 40, 50, 60, 70]
    signals = [{"timestamp": "00:02"}]  # 2s * 10fps = kare 20 -> pozisyon 2

    selected, indices = select_critical_frames(
        frames, sampled_indices, signals, fps=10.0, max_count=4
    )

    assert len(selected) == 4
    assert len(indices) == 4
    assert 20 in indices  # olay karesi mutlaka seçilmeli


def test_selection_is_deduplicated_and_chronological():
    frames = _frames(8)
    sampled_indices = [0, 10, 20, 30, 40, 50, 60, 70]
    # Aynı kareyi işaret eden iki sinyal -> tek kare seçilmeli
    signals = [{"timestamp": "00:01"}, {"timestamp": "00:01"}]

    selected, indices = select_critical_frames(
        frames, sampled_indices, signals, fps=10.0, max_count=3
    )

    assert len(indices) == len(set(indices))  # tekrar yok
    assert indices == sorted(indices)  # kronolojik
    assert len(selected) == 3


def test_empty_inputs():
    assert select_critical_frames([], [], [], fps=25.0) == ([], [])
    frames = _frames(4)
    selected, indices = select_critical_frames(frames, [0, 1, 2, 3], [], fps=25.0, max_count=0)
    assert selected == [] and indices == []


def test_max_count_caps_at_frame_count():
    frames = _frames(3)
    sampled_indices = [0, 1, 2]
    selected, indices = select_critical_frames(
        frames, sampled_indices, [], fps=25.0, max_count=10
    )
    assert len(selected) == 3
    assert len(indices) == 3
