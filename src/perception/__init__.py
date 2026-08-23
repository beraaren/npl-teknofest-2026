"""Algı katmanı (Kanal A): nesne tespiti, takip ve sahne grafiği çıkarımı.

Bu paket, video karelerinden objektif, yorumsuz gözlem verisi üreten katmandır.
Tipik kullanım akışı :class:`ObserverAgent` üzerinden gerçekleşir; alt
bileşenler (detector, tracker, scene graph) doğrudan kullanılabilir ama
genellikle bu üst seviye sınıfın içinde birleştirilmiş olarak çalışırlar.

Ana bileşenler:
  - :class:`ObserverAgent` — orkestratör; :func:`create_detector` ile seçilen
    backend'i, :class:`ObjectTracker`'ı ve :class:`SceneGraph` çıkarımını
    birleştirip kare başına gözlem sözlüğü üretir.
  - :class:`ObjectDetector` / :class:`HFObjectDetector` — tespit backend'leri
    (bkz. `detector.py`, `hf_detector.py`).
  - :class:`ObjectTracker` — Ultralytics takip (ByteTrack/BoT-SORT) wrapper'ı
    (bkz. `tracker.py`; ilgili :class:`~src.perception.tracker.TrackedObject`
    veri modeli de aynı modülde tanımlıdır).
  - :class:`SceneGraph` / :class:`SceneNode` — kare içi mekânsal ilişki
    (``near``, ``carrying``, ``wearing``) çıkarımı (bkz. `scene_graph.py`).

`vehicle_labeler.py` (`label_vehicles`, `apply_vehicle_labels`) bu paketin bir
parçası olsa da, VLM'e bağımlı olduğu için bu ``__init__`` tarafından dışa
aktarılmaz; doğrudan modül yolundan import edilir.
"""
from .detector import ObjectDetector, create_detector
from .hf_detector import HFObjectDetector
from .observer_agent import ObserverAgent
from .scene_graph import SceneGraph, SceneNode
from .tracker import ObjectTracker

__all__ = [
    "ObjectDetector",
    "HFObjectDetector",
    "create_detector",
    "ObjectTracker",
    "SceneGraph",
    "SceneNode",
    "ObserverAgent",
]
