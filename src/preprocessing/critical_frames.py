"""Kritik kare seçimi: VLM'e (Kanal B) gidecek en bilgilendirici az sayıda kare.

Önce olay sinyallerinin zaman damgalarına denk gelen kareler seçilir;
kalan kontenjan kareler arası değişim (motion) + keskinlik skoru en
yüksek karelerle doldurulur.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray


def select_critical_frames(
    frames: List[NDArray[np.uint8]],
    sampled_indices: List[int],
    event_signals: List[Dict[str, Any]],
    fps: float,
    max_count: int = 4,
) -> Tuple[List[NDArray[np.uint8]], List[int]]:
    """Kritik kareleri seçer; (kareler, gerçek_video_indeksleri) döner.

    frames: örneklenmiş kareler (sampled_indices ile aynı uzunlukta).
    """
    if not frames or max_count <= 0:
        return [], []

    max_count = min(max_count, len(frames))
    chosen: set[int] = set()  # frames listesindeki pozisyonlar

    # 1) Olay sinyallerinin timestamp'ine en yakın örneklenmiş kareler
    for sig in event_signals:
        ts = _timestamp_to_seconds(sig.get("timestamp"))
        if ts is None or not fps:
            continue
        target_frame = ts * fps
        pos = min(range(len(sampled_indices)), key=lambda p: abs(sampled_indices[p] - target_frame))
        chosen.add(pos)
        if len(chosen) >= max_count:
            break

    # 2) Kalan kontenjan: yüksek değişim + keskinlik skorlu kareler
    if len(chosen) < max_count:
        scored = []
        for p in range(len(frames)):
            if p in chosen:
                continue
            prev = frames[p - 1] if p > 0 else frames[p]
            motion = float(np.mean(np.abs(frames[p].astype(np.int16) - prev.astype(np.int16))))
            sharpness = _laplacian_variance(frames[p])
            scored.append((motion + 0.01 * sharpness, p))
        scored.sort(reverse=True)
        for _, p in scored:
            if len(chosen) >= max_count:
                break
            chosen.add(p)

    ordered = sorted(chosen)
    return [frames[p] for p in ordered], [sampled_indices[p] for p in ordered]


def _timestamp_to_seconds(ts: Any) -> float | None:
    """"MM:SS" veya "HH:MM:SS" ya da sayısal timestamp'i saniyeye çevirir."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        parts = str(ts).split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except (ValueError, TypeError):
        return None
    return None


def _laplacian_variance(frame: NDArray[np.uint8]) -> float:
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
