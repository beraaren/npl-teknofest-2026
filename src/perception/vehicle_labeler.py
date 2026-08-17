"""Araç isimlendirme: YOLO'nun genel 'arac' etiketlerini VLM ile spesifikleştirir.

Akış:
  1. 'arac' track'lerinden temsili kırpıntılar (crop) toplanır
  2. Kırpıntılar Kanal B / karar çağrılarından ÖNCE VLM'e gönderilir
  3. VLM her aracı kapalı sınıf kümesinden isimlendirir (forklift, truck, ...)
  4. Sonuç track_id -> etiket haritası olarak döner; apply_vehicle_labels()
     ile track/observation verilerine işlenir

Kural motoru (isg_rules_engine) CanonicalClass.normalize() üzerinden spesifik
isimleri yine 'arac' kanonik sınıfına indirger — etiket güncellemesi kuralları bozmaz.
VLM hatası / parse başarısızlığında sessiz fallback: etiketler 'arac' kalır.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import numpy as np
from numpy.typing import NDArray

from ..utils.logger import get_logger

logger = get_logger("VehicleLabeler")

# VLM'in seçebileceği kapalı sınıf kümesi. Küme dışı etiketler reddedilir.
VEHICLE_TYPES: List[str] = [
    "forklift", "crane", "excavator", "loader", "truck", "pickup",
    "car", "van", "bus", "motorcycle", "bicycle", "other",
]


def build_labeling_prompt(n_images: int) -> str:
    """Araç kırpıntılarını sınıflandıran VLM prompt'u üretir (İngilizce — JSON tutarlılığı için)."""
    type_list = ", ".join(VEHICLE_TYPES)
    return (
        f"You are given {n_images} cropped image(s), each showing a single vehicle or machine "
        "from a work site / surveillance camera. Classify EACH image INDEPENDENTLY.\n\n"
        "THINK STEP BY STEP before labeling:\n"
        "Consider visual cues: size, shape, wheels vs tracks, cabin position, forks, boom arm, "
        "bucket, flatbed, road context vs industrial site. A vehicle on a public road is most "
        "likely a car/truck/bus — do NOT assume it is industrial equipment.\n\n"
        f"Allowed vehicle_type values (closed set, pick exactly one): {type_list}.\n"
        "Use 'other' only when none of the specific types fit.\n\n"
        "Answer ONLY with the following JSON schema, no other text:\n"
        "{\n"
        '  "vehicles": [\n'
        '    {"image_index": 0, "vehicle_type": "forklift", "confidence_hint": "low|medium|high", "reasoning": "short step-by-step justification"}\n'
        "  ]\n"
        "}\n"
        f"image_index is 0-based and must cover every image (0-{n_images - 1})."
    )


def _extract_json(text: str) -> Dict[str, Any] | None:
    """```json fence'lerini temizleyip ilk { ... } bloğunu parse etmeyi dener."""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _padded_crop(
    frame: NDArray[np.uint8],
    bbox: tuple[float, float, float, float],
    padding_ratio: float,
) -> NDArray[np.uint8] | None:
    """BBox'ı padding_ratio oranında genişletip kareden kırpar; sınırlara kırpılır."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    ix1 = max(0, int(x1 - pad_x))
    iy1 = max(0, int(y1 - pad_y))
    ix2 = min(w_img, int(x2 + pad_x + 0.5))
    iy2 = min(h_img, int(y2 + pad_y + 0.5))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return frame[iy1:iy2, ix1:ix2]


def collect_vehicle_crops(
    tracks: Dict[int, Any],
    frames: List[NDArray[np.uint8]],
    min_confidence: float = 0.35,
    max_vehicles: int = 8,
    padding_ratio: float = 0.15,
) -> List[Dict[str, Any]]:
    """'arac' track'lerinden temsili kırpıntıları toplar.

    Temsili detection: track history'sindeki en yüksek confidence'lı (eşitlikte
    en büyük bbox alanlı) kayıt — son kare şart değil.
    Döner: [{"track_id", "crop", "frame_idx", "yolo_confidence"}], confidence'a göre azalan.
    """
    candidates: List[Dict[str, Any]] = []
    for track_id, track in tracks.items():
        if track.class_name != "arac":
            continue
        history = getattr(track, "history", None) or [track.last_detection]
        best = max(
            history,
            key=lambda d: (d.confidence, d.width * d.height),
        )
        if best.confidence < min_confidence:
            continue
        if not (0 <= best.frame_idx < len(frames)):
            continue
        crop = _padded_crop(frames[best.frame_idx], best.bbox, padding_ratio)
        if crop is None:
            continue
        candidates.append(
            {
                "track_id": track_id,
                "crop": crop,
                "frame_idx": best.frame_idx,
                "yolo_confidence": best.confidence,
            }
        )

    candidates.sort(key=lambda c: c["yolo_confidence"], reverse=True)
    return candidates[:max_vehicles]


def label_vehicles(
    tracks: Dict[int, Any],
    frames: List[NDArray[np.uint8]],
    backend: Any,
    config: Any,
) -> Dict[int, Dict[str, Any]]:
    """Araç track'lerini VLM ile isimlendirir.

    Döner: {track_id: {"vehicle_type", "confidence_hint", "reasoning"}}
    Herhangi bir hata / parse başarısızlığında {} döner (sessiz fallback —
    etiketler 'arac' kalır, pipeline eskisi gibi çalışır).
    """
    if not getattr(config, "enabled", False):
        return {}

    candidates = collect_vehicle_crops(
        tracks,
        frames,
        min_confidence=getattr(config, "min_confidence", 0.35),
        max_vehicles=getattr(config, "max_vehicles", 8),
        padding_ratio=getattr(config, "padding_ratio", 0.15),
    )
    if not candidates:
        return {}

    prompt = build_labeling_prompt(len(candidates))
    try:
        raw = backend.generate(
            [c["crop"] for c in candidates],
            prompt,
            max_tokens=getattr(config, "max_tokens", 768),
        )
    except Exception as exc:  # VLM çökmesi tüm pipeline'ı durdurmamalı
        logger.warning(f"Araç isimlendirme VLM çağrısı başarısız ({exc}); 'arac' etiketleri korunuyor.")
        return {}

    parsed = _extract_json(raw or "")
    if not parsed or not isinstance(parsed.get("vehicles"), list):
        logger.warning("Araç isimlendirme çıktısı parse edilemedi; 'arac' etiketleri korunuyor.")
        return {}

    label_map: Dict[int, Dict[str, Any]] = {}
    for entry in parsed["vehicles"]:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("image_index")
        vtype = str(entry.get("vehicle_type", "")).lower().strip()
        if not isinstance(idx, int) or not (0 <= idx < len(candidates)):
            continue
        if vtype not in VEHICLE_TYPES:
            continue
        label_map[candidates[idx]["track_id"]] = {
            "vehicle_type": vtype,
            "confidence_hint": entry.get("confidence_hint", "low"),
            "reasoning": entry.get("reasoning", ""),
        }

    logger.info(
        f"Araç isimlendirme: {len(label_map)}/{len(candidates)} araç etiketlendi "
        f"({sorted({v['vehicle_type'] for v in label_map.values()})})"
    )
    return label_map


def apply_vehicle_labels(
    tracks: Dict[int, Any],
    observations: List[Dict[str, Any]],
    label_map: Dict[int, Dict[str, Any]],
) -> int:
    """İsimlendirme sonucunu track ve observation verilerine işler.

    Güncellenen yerler (EventEngine üçünü de okur — tutarlılık şart):
      - TrackedObject.class_name + history'deki Detection'ların class_name'i
      - observation["detections"][*]["class"]   (track_id eşleşmesi)
      - observation["tracks"][*]["class"]       (track_id eşleşmesi)
      - observation["scene_graph"]["nodes"][*]["class"] ve "node_id"
        (node_id '{class}_{track_id}' formatında üretiliyor)

    Döner: güncellenen track sayısı.
    """
    if not label_map:
        return 0

    # 1. Canlı track nesneleri
    for track_id, info in label_map.items():
        track = tracks.get(track_id)
        if track is None:
            continue
        track.class_name = info["vehicle_type"]
        for det in track.history:
            if det.class_name == "arac":
                det.class_name = info["vehicle_type"]

    # 2. Observation dict'leri (EventEngine bunları tüketir)
    for obs in observations:
        for det in obs.get("detections", []):
            info = label_map.get(det.get("track_id"))
            if info and det.get("class") == "arac":
                det["class"] = info["vehicle_type"]
        for tr in obs.get("tracks", []):
            info = label_map.get(tr.get("track_id"))
            if info and tr.get("class") == "arac":
                tr["class"] = info["vehicle_type"]
        for node in obs.get("scene_graph", {}).get("nodes", []):
            node_id = node.get("node_id", "")
            if node.get("class") != "arac" or "_" not in node_id:
                continue
            try:
                tid = int(node_id.rsplit("_", 1)[1])
            except ValueError:
                continue
            info = label_map.get(tid)
            if info:
                node["class"] = info["vehicle_type"]
                node["node_id"] = f"{info['vehicle_type']}_{tid}"

    return len(label_map)
