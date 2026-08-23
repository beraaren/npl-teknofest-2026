"""HuggingFace transformers tabanlı geçici nesne tespit backend'i.

Bu modül, projenin özel İSG YOLO modeli eğitilene/entegre edilene kadar
kullanılan bir **yer tutucu** (placeholder) tespit backend'i sağlar. Varsayılan
model (``PaddlePaddle/PP-DocLayoutV3_safetensors``) bir doküman düzeni analizi
modelidir ve İSG sahnelerinde (insan, forklift, KKD) anlamlı tespit üretmesi
beklenmez; amacı, algı katmanının arayüzünü (:class:`Detection` sözleşmesi)
gerçek model gelene kadar uçtan uca test edilebilir kılmaktır.

Model kimliği ``config.yaml`` içindeki ``perception.hf_model`` anahtarı ile
değiştirilebilir; bu backend :func:`src.perception.detector.create_detector`
factory'si üzerinden ``detector_backend: "hf_transformers"`` seçildiğinde
devreye girer.

İlgili not: `benioku.md` §5 — "PP-DocLayoutV3 geçicidir; üretime dair tespit
bekleme." Gerçek İSG YOLO'su hazır olduğunda ``detector_backend: "ultralytics"``
değerine dönülmesi planlanmaktadır.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..utils.logger import get_logger
from .detector import Detection


class HFObjectDetector:
    """HuggingFace ``AutoImageProcessor`` + ``AutoModelForObjectDetection`` wrapper'ı.

    :class:`src.perception.detector.ObjectDetector` (Ultralytics) ile aynı
    ``detect`` / ``detect_batch`` arayüzünü sağlar, böylece
    :class:`src.perception.observer_agent.ObserverAgent` iki backend'i
    ayırt etmeden kullanabilir. Tek fark :attr:`supports_tracking` bayrağıdır.

    Model yükleme lazy'dir (:meth:`_load`): ``torch`` ve ``transformers``
    bağımlılıkları yalnızca ilk gerçek çağrıda import edilir.

    Attributes:
        supports_tracking: ``False``. HuggingFace ``AutoModelForObjectDetection``
            modelleri Ultralytics'in ``model.track()`` metoduna sahip değildir;
            bu bayrak sayesinde
            :meth:`src.perception.observer_agent.ObserverAgent._track_by_iou`
            devreye girip takibi IoU eşleşmesiyle kendisi yürütür.
        model_path: HuggingFace model kimliği veya yerel yol.
        confidence: Tespitlerin kabul edileceği minimum güven eşiği.
        custom_classes: Boş değilse, eşlenmiş sınıf adı bu kümede değilse
            eşleme geri alınır (bkz. :meth:`_map_class`).
        device: Çalıştırma cihazı tercihi (``"auto"`` | ``"cuda"`` | ``"cpu"``).
            ``"auto"`` seçiliyse, model yüklenirken CUDA varlığına göre karar
            verilir ve gerçek seçim :attr:`_device`'ta saklanır.
    """

    supports_tracking = False

    def __init__(
        self,
        model_path: str = "PaddlePaddle/PP-DocLayoutV3_safetensors",
        confidence: float = 0.5,
        custom_classes: List[str] | None = None,
        device: str = "auto",
    ):
        """Detector'ı yapılandırır; model henüz yüklenmez (lazy).

        Args:
            model_path: HuggingFace model kimliği veya yerel yol.
            confidence: Tespit kabul eşiği (0.0-1.0).
            custom_classes: Eşlenmiş sınıf adlarının kısıtlanacağı küme.
            device: ``"auto"`` | ``"cuda"`` | ``"cpu"``.
        """
        self.model_path = model_path
        self.confidence = confidence
        self.custom_classes = set(custom_classes or [])
        self.device = device  # "auto" | "cuda" | "cpu"
        self.logger = get_logger("HFObjectDetector")
        self._model = None
        self._processor = None
        self._device = "cpu"

    def _load(self):
        """Model ve processor'ı ilk çağrıda yükler, cihazı çözümler ve önbelleğe alır.

        ``device="auto"`` verildiyse, CUDA kullanılabilirliğine göre gerçek
        çalıştırma cihazı (:attr:`_device`) burada belirlenir ve bir daha
        değişmez; bu, aynı model örneğinin karışık cihazlarda tensor
        hatalarıyla karşılaşmasını önler.

        Returns:
            Yüklü ve ``eval()`` moduna alınmış model nesnesi.
        """
        if self._model is None:
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection

            if self.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.device
            self._processor = AutoImageProcessor.from_pretrained(self.model_path)
            self._model = AutoModelForObjectDetection.from_pretrained(self.model_path).to(self._device)
            self._model.eval()
            self.logger.info(f"HF detection modeli yüklendi: {self.model_path} ({self._device})")
        return self._model

    def detect(self, frame: NDArray[np.uint8], frame_idx: int = 0) -> List[Detection]:
        """Tek bir karede nesne tespiti yapar (takipsiz).

        Çıktı tensor'ları, post-processing adımından önce bilinçli olarak
        CPU'ya taşınır (bkz. kod içi not): bazı model/transformers sürüm
        kombinasyonlarında (PP-DocLayoutV3 + transformers 5.x) GPU üzerinde
        post-processing çağrısı "illegal memory access" hatası vermektedir.
        Bu geçici bir uyumluluk önlemidir, performans optimizasyonu değildir.

        Args:
            frame: RGB, ``HWC`` düzeninde ``uint8`` kare.
            frame_idx: Bu karenin indeksi; sonuç :class:`Detection`
                kayıtlarına aktarılır.

        Returns:
            Karede bulunan her nesne için bir :class:`Detection`. Model
            poligon çıktısı üretiyorsa (segmentasyon), ilgili
            :attr:`Detection.polygon` alanı da doldurulur.
        """
        import torch

        model = self._load()
        image = Image.fromarray(frame)
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        # Post-processing GPU'da bazı modellerde (PP-DocLayoutV3 + transformers 5.x)
        # illegal memory access veriyor; tensor'ları CPU'ya taşı (tip korunarak).
        outputs = type(outputs)(
            {k: v.cpu() if torch.is_tensor(v) else v for k, v in outputs.items()}
        )
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
        """Ham model sınıf adını kod tabanının kanonik (Türkçe) adına çevirir.

        Bu sınıfın eşleme sözlüğü, :meth:`src.perception.detector.ObjectDetector._map_class`
        ile **aynı değildir** — HF backend'i geçici olduğu için burada
        (``"kamyon"``, ``"araba"``) gibi daha kaba/geçici Türkçe adlar kullanılır.
        Gerçek YOLO backend'ine geçildiğinde bu eşleme kümesinin de gözden
        geçirilmesi gerekir.

        Args:
            class_name: Modelin döndürdüğü ham sınıf adı.

        Returns:
            Eşleme kümesindeyse kanonik ad; aksi halde (veya
            ``custom_classes`` kısıtlamasını geçemiyorsa) orijinal ad.
        """
        mapping = {
            "person": "insan",
            "truck": "kamyon",
            "car": "araba",
            "pallet": "palet",
        }
        mapped = mapping.get(class_name.lower(), class_name)
        if self.custom_classes and mapped not in self.custom_classes:
            return class_name
        return mapped

    def detect_batch(self, frames: List[NDArray[np.uint8]]) -> List[List[Detection]]:
        """Birden fazla kareyi sırayla tespit eder.

        Args:
            frames: RGB kare listesi.

        Returns:
            Her kare için :meth:`detect` çıktısı; kare indeksi listedeki
            konuma göre atanır.
        """
        return [self.detect(f, idx) for idx, f in enumerate(frames)]
