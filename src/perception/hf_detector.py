"""HuggingFace transformers tabanlı nesne tespit backend'i.

YOLO fine-tune edilene kadar geçici backend olarak kullanılır
(örn. PaddlePaddle/PP-DocLayoutV3_safetensors). Model kimliği
config.yaml -> perception.hf_model üzerinden değiştirilebilir.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..utils.logger import get_logger
from .detector import Detection


class HFObjectDetector:
    """AutoImageProcessor + AutoModelForObjectDetection wrapper'ı.

    Not: Ultralytics dışı modeller `.track()` desteklemez; takip
    ObserverAgent tarafında IoU eşleşmesiyle yapılır.
    """

    supports_tracking = False

    def __init__(
        self,
        model_path: str = "PaddlePaddle/PP-DocLayoutV3_safetensors",
        confidence: float = 0.5,
        custom_classes: List[str] | None = None,
    ):
        self.model_path = model_path
        self.confidence = confidence
        self.custom_classes = set(custom_classes or [])
        self.logger = get_logger("HFObjectDetector")
        self._model = None
        self._processor = None
        self._device = "cpu"

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._processor = AutoImageProcessor.from_pretrained(self.model_path)
            self._model = AutoModelForObjectDetection.from_pretrained(self.model_path).to(self._device)
            self._model.eval()
            self.logger.info(f"HF detection modeli yüklendi: {self.model_path} ({self._device})")
        return self._model

    def detect(self, frame: NDArray[np.uint8], frame_idx: int = 0) -> List[Detection]:
        import torch

        model = self._load()
        image = Image.fromarray(frame)
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        results = self._processor.post_process_object_detection(
            outputs,
            threshold=self.confidence,
            target_sizes=[image.size[::-1]],
        )

        id2label = getattr(model.config, "id2label", {}) or {}
        detections: List[Detection] = []
        for result in results:
            # Standart detection modelleri polygon döndürmez; varsa sakla
            polygons = result.get("polygon_points")
            for idx, (score, label_id, box) in enumerate(
                zip(result["scores"], result["labels"], result["boxes"])
            ):
                label = int(label_id.item() if hasattr(label_id, "item") else label_id)
                class_name = self._map_class(id2label.get(label, str(label)))
                det = Detection(
                    class_name=class_name,
                    confidence=float(score.item() if hasattr(score, "item") else score),
                    bbox=tuple(float(v) for v in box.tolist()),
                    frame_idx=frame_idx,
                )
                if polygons is not None and idx < len(polygons):
                    pts = polygons[idx]
                    det.polygon = pts.tolist() if hasattr(pts, "tolist") else pts
                detections.append(det)
        return detections

    def _map_class(self, class_name: str) -> str:
        mapping = {
            "person": "insan",
            "truck": "forklift",
            "car": "forklift",
            "pallet": "palet",
        }
        mapped = mapping.get(class_name.lower(), class_name)
        if self.custom_classes and mapped not in self.custom_classes:
            return class_name
        return mapped

    def detect_batch(self, frames: List[NDArray[np.uint8]]) -> List[List[Detection]]:
        return [self.detect(f, idx) for idx, f in enumerate(frames)]
