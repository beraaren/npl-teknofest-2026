"""Nesne tespit wrapper'ı ve algılama sonucu veri modeli.

Bu modül, Kanal A algı hattının ilk adımını oluşturur: bir video karesini alıp
üzerindeki nesneleri sınıf adı, güven skoru ve sınırlayıcı kutu (bbox) ile
raporlayan :class:`Detection` veri modelini ve bunu üreten
:class:`ObjectDetector` (Ultralytics YOLO) wrapper'ını sağlar.

İki farklı tespit backend'i desteklenir (:func:`create_detector` factory'si
üzerinden seçilir):

``ultralytics``
    Üretimde kullanılan asıl backend. Ultralytics YOLO modeli ile çalışır,
    hem tespit hem de yerleşik takip (``model.track()``) sağlar
    (bkz. :attr:`ObjectDetector.supports_tracking`).

``hf_transformers``
    :mod:`src.perception.hf_detector` içindeki geçici backend. YOLO modeli
    İSG sahneleri üzerinde eğitilene kadar kullanılan bir yer tutucudur;
    takip desteği yoktur.

Sınıf adı eşlemesi
-------------------
Modelin döndürdüğü ham sınıf adları (COCO etiketleri veya özel eğitim
sınıfları), kod tabanının kullandığı Türkçe kanonik sınıf adlarına
:meth:`ObjectDetector._map_class` üzerinden çevrilir (örn. ``"person"`` →
``"insan"``, ``"forklift"`` → ``"arac"``). Bu eşleme, algı katmanından sonraki
tüm modüllerin (kural motoru, sahne grafiği, RAG) sabit bir Türkçe sınıf
kümesiyle çalışabilmesini sağlar; model değişse (COCO, özel İSG YOLO'su) dahi
aşağı akış kodunun değişmemesi gerekir.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np
from numpy.typing import NDArray


class Detection:
    """Bir karede algılanan tek bir nesneyi temsil eden sonuç kaydı.

    Bu sınıf, tüm algı katmanının (tracker, scene graph, olay kuralları)
    ortak veri birimidir. Piksel uzayındaki geometriden türetilen
    :attr:`center`, :attr:`width`, :attr:`height` ve :attr:`aspect_ratio`
    özellikleri, kural motorunun (`src/events/rules.py`) ölçek-bağımsız
    hesaplarında (örn. düşme/devrilme tespiti) doğrudan kullanılır.

    Attributes:
        class_name: Kanonik (Türkçe) sınıf adı, örn. ``"insan"``, ``"arac"``,
            ``"baret"``. Ham model çıktısı :meth:`ObjectDetector._map_class`
            ile bu forma çevrilmiştir.
        confidence: Modelin algılama güven skoru, 0.0-1.0 aralığında.
        bbox: Piksel uzayında ``(x1, y1, x2, y2)`` sol-üst / sağ-alt
            sınırlayıcı kutu koordinatları.
        frame_idx: Bu algılamanın ait olduğu kare indeksi.
        track_id: Nesnenin video boyunca korunan takip kimliği. Takip
            yapılmamışsa (tek kare tespiti) ``None``.
        polygon: Segmentasyon çıktısı üreten modellerde (örn.
            PP-DocLayoutV3) poligon noktaları; standart tespit modellerinde
            ``None`` kalır.
    """

    def __init__(
        self,
        class_name: str,
        confidence: float,
        bbox: tuple[float, float, float, float],
        frame_idx: int = 0,
        track_id: int | None = None,
    ):
        """Yeni bir algılama kaydı oluşturur.

        Args:
            class_name: Kanonik sınıf adı.
            confidence: Güven skoru (0.0-1.0).
            bbox: ``(x1, y1, x2, y2)`` piksel koordinatları.
            frame_idx: Algılamanın ait olduğu kare indeksi.
            track_id: Varsa takip kimliği; yoksa ``None``.
        """
        self.class_name = class_name
        self.confidence = confidence
        # x1, y1, x2, y2 (piksel)
        self.bbox = bbox
        self.frame_idx = frame_idx
        self.track_id = track_id
        # Bazı modeller (örn. PP-DocLayoutV3) segmentasyon poligonu da döndürür
        self.polygon: Any = None

    @property
    def center(self) -> tuple[float, float]:
        """Sınırlayıcı kutunun geometrik merkezi.

        Returns:
            ``(cx, cy)`` biçiminde merkez koordinatı, piksel uzayında.
        """
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        """Sınırlayıcı kutunun piksel cinsinden genişliği."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """Sınırlayıcı kutunun piksel cinsinden yüksekliği.

        Kural motorunda (`src/events/state_machine.py`) bu değerin zaman
        içindeki üstel ortalaması (``scale_ema``), nesnenin kameraya
        uzaklığının dolaylı bir göstergesi olarak kullanılır — sabit piksel
        eşikleri yerine bu ölçeğe oranlı eşiklerle çalışmak, kural
        sonuçlarını kameraya uzaklıktan bağımsız kılar.
        """
        return self.bbox[3] - self.bbox[1]

    @property
    def aspect_ratio(self) -> float:
        """Sınırlayıcı kutunun genişlik/yükseklik oranı.

        Forklift devrilmesi gibi kuralların ana geometrik göstergesidir:
        dikey duran bir araçta oran düşük, yan yatmış veya yandan görünen bir
        araçta oran yüksektir. Tek başına devrilme kanıtı sayılmaz (araç
        kameraya doğru dönerken de oran benzer şekilde artar); kural motoru
        bunu ek bir kanıtla (yükseklik çöküşü) birleştirir.

        Returns:
            ``width / height``. Yükseklik sıfırsa (dejenere kutu) ``0.0``
            döner; sıfıra bölme hatası oluşmaz.
        """
        h = self.height
        return self.width / h if h > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Algılamayı JSON uyumlu bir sözlüğe dönüştürür.

        Bu sözlük biçimi, gözlem (observation) çıktısında, VLM prompt'una
        aktarılan bağlamda ve nihai analiz raporunda kullanılır.

        Returns:
            ``class``, ``track_id``, ``confidence``, ``bbox``, ``center``,
            ``frame_idx`` anahtarlarını içeren sözlük. Poligon varsa
            ``polygon_points`` anahtarı da eklenir. Sayısal değerler
            raporlama için yuvarlanır.
        """
        data = {
            "class": self.class_name,
            "track_id": self.track_id,
            "confidence": round(self.confidence, 3),
            "bbox": [round(v, 2) for v in self.bbox],
            "center": [round(self.center[0], 2), round(self.center[1], 2)],
            "frame_idx": self.frame_idx,
        }
        if self.polygon is not None:
            data["polygon_points"] = self.polygon
        return data


class ObjectDetector:
    """Ultralytics YOLOv8 tabanlı nesne tespit ve sınıf-eşleme wrapper'ı.

    Bu sınıf, projenin varsayılan ve üretimde kullanılan tespit backend'idir.
    Model yükleme lazy'dir (:meth:`_load`): ağır ``ultralytics`` bağımlılığı
    yalnızca ilk gerçek çağrıda import edilir, böylece testler ve ``--help``
    gibi model gerektirmeyen komutlar bağımlılık kurulmadan çalışabilir.

    Attributes:
        supports_tracking: ``True``. Bu bayrak, :class:`ObjectTracker`'ın
            Ultralytics'in yerleşik ``model.track()`` metodunu kullanabileceğini
            gösterir; ``False`` olan backend'lerde (örn.
            :class:`src.perception.hf_detector.HFObjectDetector`) takip
            :meth:`src.perception.observer_agent.ObserverAgent._track_by_iou`
            ile IoU eşleşmesi üzerinden ayrıca yapılır.
        model_path: Ultralytics model dosyasının yolu veya adı (örn.
            ``"yolov8n.pt"`` ya da özel eğitilmiş bir ``.pt`` dosyası).
        confidence: Tespitlerin kabul edileceği minimum güven eşiği.
        custom_classes: Boş değilse, eşlenmiş sınıf adı bu kümede değilse
            eşleme geri alınır (bkz. :meth:`_map_class`).
    """

    supports_tracking = True

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.35, custom_classes: List[str] | None = None, device: Any = None):
        """Detector'ı yapılandırır; model henüz yüklenmez (lazy).

        Args:
            model_path: Ultralytics model dosyasının yolu veya adı.
            confidence: Tespit kabul eşiği (0.0-1.0).
            custom_classes: Eşlenmiş sınıf adlarının kısıtlanacağı küme.
                ``None`` veya boşsa kısıtlama uygulanmaz.
            device: 'cuda', 0 veya 'cpu'. None ise otomatik GPU/CPU seçilir.
        """
        import torch
        self.model_path = model_path
        self.confidence = confidence
        self.custom_classes = set(custom_classes or [])
        if device is None:
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._model = None

    def _load(self):
        """Ultralytics ``YOLO`` modelini ilk çağrıda yükler ve önbelleğe alır.

        Returns:
            Yüklü ``ultralytics.YOLO`` model nesnesi.
        """
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            if self.device != "cpu":
                try:
                    self._model.to(self.device)
                except Exception:
                    pass
        return self._model

    def detect(self, frame: NDArray[np.uint8], frame_idx: int = 0) -> List[Detection]:
        """Tek bir karede nesne tespiti yapar (takipsiz).

        Args:
            frame: RGB, ``HWC`` düzeninde ``uint8`` kare.
            frame_idx: Bu karenin indeksi; sonuç :class:`Detection`
                kayıtlarına aktarılır.

        Returns:
            Karede bulunan her nesne için bir :class:`Detection`. Hiç tespit
            yoksa boş liste döner.
        """
        model = self._load()
        results = model(frame, verbose=False, conf=self.confidence, device=self.device)[0]
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

    def _map_class(self, class_name: str) -> str:
        """Ham model sınıf adını kod tabanının kanonik (Türkçe) adına çevirir.

        Eşleme, modelin ürettiği ham etiketten (küçük harfe çevrilerek)
        Türkçe kanonik ada bakan sabit bir sözlükle yapılır. ``custom_classes``
        boş değilse ve eşlenmiş ad bu kümede yoksa, eşleme geri alınır ve
        orijinal (ham) ad döndürülür — bu, belirli bir sınıf kümesiyle
        sınırlı çalışan yapılandırmalarda beklenmeyen sınıfların sessizce
        yanlış bir kanonik ada düşmesini önler.

        Args:
            class_name: Modelin döndürdüğü ham sınıf adı.

        Returns:
            Eşleme kümesindeyse kanonik Türkçe ad; aksi halde (veya
            ``custom_classes`` kısıtlamasını geçemiyorsa) orijinal ad.
        """
        mapping = {
            "person": "insan",
            "forklift": "arac",
            "car": "arac",
            "machinery": "arac",
            "pallet": "palet",
            "helmet": "baret",
            "vest": "yelek",
            "fire": "yangin",
            "smoke": "duman",
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


def create_detector(config: Any) -> Any:
    """`config.perception`'a göre uygun tespit backend'ini oluşturan factory.

    Bu fonksiyon, algı katmanının tek giriş noktasıdır; çağıranlar
    (:class:`src.perception.observer_agent.ObserverAgent`, `src/main.py`)
    hangi backend'in kullanıldığını bilmeden aynı arayüzle (`detect`,
    `detect_batch`, `supports_tracking`) çalışır. Yeni bir backend eklemek
    isteyen geliştiriciler bu fonksiyona bir dal eklemelidir.

    ``detector_backend`` değerine göre seçim:

      - ``"ultralytics"`` → :class:`ObjectDetector` (YOLO, ByteTrack destekli).
        Varsayılan ve üretim backend'i.
      - ``"hf_transformers"`` → :class:`src.perception.hf_detector.HFObjectDetector`
        (geçici; özel İSG YOLO modeli eğitilene kadar kullanılan yer tutucu).

    Args:
        config: `PerceptionConfig` örneği (veya aynı alanlara sahip herhangi
            bir nesne): `detector_backend`, `yolo_model`, `confidence_threshold`,
            `custom_classes` ve HF backend'i için `hf_model`, `hf_threshold`,
            `hf_device` alanlarını okur.

    Returns:
        Seçilen backend'in örneği (:class:`ObjectDetector` veya
        :class:`~src.perception.hf_detector.HFObjectDetector`).
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
            device=getattr(config, "hf_device", "auto"),
        )

    if backend != "ultralytics":
        logger.warning(f"Bilinmeyen detector_backend '{backend}'; ultralytics kullanılacak.")

    return ObjectDetector(
        model_path=config.yolo_model,
        confidence=config.confidence_threshold,
        custom_classes=config.custom_classes,
    )
