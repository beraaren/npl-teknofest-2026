"""Nesne tespit wrapper'ı."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np
from numpy.typing import NDArray


class Detection:
    """Tek bir tespit sonucu."""

    def __init__(
        self,
        class_name: str,
        confidence: float,
        bbox: tuple[float, float, float, float],
        frame_idx: int = 0,
    ):
        self.class_name = class_name
        self.confidence = confidence
        # x1, y1, x2, y2 (piksel)
        self.bbox = bbox
        self.frame_idx = frame_idx
        # Bazı modeller (örn. PP-DocLayoutV3) segmentasyon poligonu da döndürür
        self.polygon: Any = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def aspect_ratio(self) -> float:
        h = self.height
        return self.width / h if h > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        data = {
            "class": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": [round(v, 2) for v in self.bbox],
            "center": [round(self.center[0], 2), round(self.center[1], 2)],
            "frame_idx": self.frame_idx,
        }
        if self.polygon is not None:
            data["polygon_points"] = self.polygon
        return data


class ObjectDetector:
    """Ultralytics YOLOv8 wrapper."""

    supports_tracking = True

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.35, custom_classes: List[str] | None = None):
        self.model_path = model_path
        self.confidence = confidence
        self.custom_classes = set(custom_classes or [])
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, frame: NDArray[np.uint8], frame_idx: int = 0) -> List[Detection]:
        model = self._load()
        results = model(frame, verbose=False, conf=self.confidence)[0]
        detections: List[Detection] = []

        if results.boxes is None:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        names = results.names

        for box, conf, cls_idx in zip(boxes, confs, classes):
            class_name = names.get(cls_idx, str(cls_idx))
            # Özel sınıf eşleme: COCO 'person' -> 'insan', 'truck' -> 'forklift' vb.
            class_name = self._map_class(class_name)
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=float(conf),
                    bbox=tuple(float(v) for v in box),
                    frame_idx=frame_idx,
                )
            )
        return detections

    def _map_class(self, coco_name: str) -> str:
        mapping = {
            "person": "insan",
            "truck": "forklift",
            "car": "forklift",
            "pallet": "palet",
        }
        mapped = mapping.get(coco_name.lower(), coco_name)
        # Eğer özel sınıf listesi varsa ve mapped içindeyse koru
        if self.custom_classes and mapped not in self.custom_classes:
            return coco_name
        return mapped

    def detect_batch(self, frames: List[NDArray[np.uint8]]) -> List[List[Detection]]:
        return [self.detect(f, idx) for idx, f in enumerate(frames)]


def create_detector(config: Any) -> Any:
    """config.perception'a göre uygun tespit backend'ini oluşturur.

    detector_backend:
      - "ultralytics"    -> ObjectDetector (YOLO, ByteTrack destekli)
      - "hf_transformers" -> HFObjectDetector (geçici; YOLO eğitilene kadar)
    """
    from ..utils.logger import get_logger

    logger = get_logger("DetectorFactory")
    backend = getattr(config, "detector_backend", "ultralytics")

    if backend == "hf_transformers":
        from .hf_detector import HFObjectDetector

        logger.info(f"HF transformers detection backend'i seçildi: {config.hf_model}")
        return HFObjectDetector(
            model_path=config.hf_model,
            confidence=getattr(config, "hf_threshold", 0.5),
            custom_classes=config.custom_classes,
        )

    if backend != "ultralytics":
        logger.warning(f"Bilinmeyen detector_backend '{backend}'; ultralytics kullanılacak.")

    return ObjectDetector(
        model_path=config.yolo_model,
        confidence=config.confidence_threshold,
        custom_classes=config.custom_classes,
    )
