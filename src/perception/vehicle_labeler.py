"""Araç isimlendirme: YOLO'nun genel 'arac' etiketlerini VLM ile spesifikleştirir.

Bu modül, algı katmanı ile karar katmanı arasında çalışan bir **zenginleştirme**
(enrichment) adımıdır. YOLO tespit modeli tüm motorlu/tekerlekli araçları tek
bir kanonik sınıfa (``"arac"``) eşler (bkz. `detector.py` ``_map_class``); bu
modül ise VLM'e bu araçların kırpılmış görüntülerini göstererek her birinin
gerçek türünü (forklift, kamyon, binek araç...) kapalı bir sınıf kümesinden
seçtirir ve sonucu geri track/observation verilerine işler.

Akış:
  1. :func:`collect_vehicle_crops` — ``"arac"`` sınıfındaki track'lerden,
     her track için en güvenilir (en yüksek confidence, eşitlikte en büyük
     bbox) algılamayı temsilci seçip kareden kırpar.
  2. :func:`build_labeling_prompt` ile hazırlanan İngilizce prompt ve
     kırpıntılar, Kanal B / karar çağrılarından **önce** VLM'e gönderilir
     (:func:`label_vehicles`).
  3. VLM her aracı :data:`VEHICLE_TYPES` kapalı kümesinden isimlendirir;
     çıktı ``{"vehicles": [{"image_index", "vehicle_type", ...}]}`` şemasına
     zorlanır ve :func:`_extract_json` ile ayrıştırılır.
  4. Sonuç ``track_id -> etiket bilgisi`` haritası olarak döner;
     :func:`apply_vehicle_labels` bu haritayı canlı track nesnelerine ve
     zaten üretilmiş observation sözlüklerine işler.

Neden bu adım kural motorunu bozmaz: `src/events/isg_rules_engine.py`
içindeki ``CanonicalClass.normalize()`` (ve bu kod tabanındaki sınıf eşleme
mantığı), spesifik isimleri (``"forklift"`` gibi) yine ``"arac"`` kanonik
sınıfına indirger. Yani isimlendirme sadece **raporlama ve VLM bağlamını**
zenginleştirir; geometrik kuralların (`src/events/rules.py`) sınıf tabanlı
karşılaştırmalarını etkilemez.

Hata toleransı: VLM çağrısı başarısız olursa veya çıktı beklenen şemaya
uymazsa, bu modül **sessizce** boş bir harita döner — etiketler ``"arac"``
olarak kalır ve pipeline'ın devamı hiç etkilenmeden çalışır. Bu bilinçli bir
tasarım kararıdır: araç isimlendirme bir "nice to have" zenginleştirmedir,
ana risk tespit akışının bir ön koşulu değildir.
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
    """Araç kırpıntılarını sınıflandıran VLM prompt'unu oluşturur.

    Prompt bilinçli olarak İngilizce yazılmıştır: JSON çıktı şemasına
    uyumun ve model tutarlılığının, İngilizce talimatlarla daha güvenilir
    olduğu gözlemlenmiştir (bu, kod tabanındaki diğer Türkçe-öncelikli
    konvansiyondan bilinçli bir sapmadır).

    Args:
        n_images: VLM'e gönderilecek kırpıntı sayısı; her birinin bir
            ``image_index`` ile (0'dan başlayarak) etiketlenmesi istenir.

    Returns:
        VLM'e gönderilecek tam prompt metni. Model, kapalı
        :data:`VEHICLE_TYPES` kümesinden birer tür seçip belirtilen JSON
        şemasında yanıt vermeye yönlendirilir.
    """
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
    """VLM'in ham metin çıktısından JSON nesnesini ayrıştırmayı dener.

    Modeller genellikle çıktıyı Markdown kod bloğuna (` ```json ... ``` `)
    sarar veya JSON'dan önce/sonra ek açıklama metni ekler; bu fonksiyon
    kod bloğu işaretlerini temizler ve metindeki ilk ``{`` ile son ``}``
    arasındaki bölümü ayrıştırır.

    Args:
        text: VLM'in ham (post-processing yapılmamış) metin çıktısı.

    Returns:
        Ayrıştırılan JSON sözlüğü. Metin boşsa, süslü parantez çifti
        bulunamazsa veya JSON geçersizse ``None``.
    """
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
    """Bir sınırlayıcı kutuyu genişletip kareden kırpar.

    Genişletme (padding), VLM'in aracın çevresindeki bağlamı (tekerlek,
    yol/tesis zemini gibi) görebilmesi için eklenir — tam bbox'a sıkı sıkıya
    kırpılmış görüntüler genellikle sınıflandırma için yetersiz bağlam
    içerir.

    Args:
        frame: Kırpılacak RGB kare.
        bbox: ``(x1, y1, x2, y2)`` piksel koordinatları.
        padding_ratio: Her kenara, o kenarın uzunluğunun bu oranı kadar
            piksel eklenir (örn. 0.15 = %15 genişletme).

    Returns:
        Kırpılmış görüntü dizisi. Genişletilmiş kutu kare sınırlarına
        kırpıldıktan sonra dejenere (genişlik veya yükseklik <= 0) hale
        geliyorsa ``None``.
    """
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
    """``"arac"`` sınıfındaki track'lerden VLM'e gönderilecek temsili kırpıntıları toplar.

    Her track için **tek** bir temsili algılama seçilir: track geçmişindeki
    en yüksek güven skoruna sahip kayıt (eşitlik durumunda en büyük bbox
    alanına sahip olan). Bu seçim son kareye bağlı değildir — bir aracın en
    net görüldüğü an, videonun sonu olmak zorunda değildir; en güvenilir
    algılamayı seçmek, VLM'e daha kaliteli bir görüntü sunar.

    Args:
        tracks: ``{track_id: TrackedObject}`` biçiminde takip durumu
            (tipik olarak :attr:`src.perception.observer_agent.ObserverAgent.tracks`).
        frames: Kırpıntıların alınacağı kare listesi; her track'in temsili
            algılamasındaki ``frame_idx`` bu listedeki bir indekse karşılık
            gelmelidir.
        min_confidence: Bu eşiğin altındaki temsili algılamaya sahip
            track'ler tamamen elenir (isimlendirmeye gönderilmez).
        max_vehicles: Döndürülecek maksimum aday sayısı (VLM çağrısının
            token/görüntü sayısını sınırlamak için).
        padding_ratio: :func:`_padded_crop`'a geçirilen genişletme oranı.

    Returns:
        Her biri ``{"track_id", "crop", "frame_idx", "yolo_confidence"}``
        anahtarlarını içeren aday sözlükleri, güven skoruna göre **azalan**
        sırada ve en fazla ``max_vehicles`` eleman.
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
    """Araç track'lerini VLM'e sorup gerçek araç türlerini belirler.

    `src/main.py` içinde, Kanal B çağrısından ve karar ajanından **önce**
    çalıştırılır; böylece isimlendirilen etiketler sonraki tüm adımlara
    (VLM bağlamı, nihai rapor) yansır.

    Hata toleransı: yapılandırma kapalıysa, hiçbir aday bulunamazsa, VLM
    çağrısı istisna fırlatırsa veya çıktı beklenen şemaya uymuyorsa, bu
    fonksiyon **istisna fırlatmadan** boş bir sözlük döner. Bu, VLM
    backend'inin kullanılamadığı veya modelin beklenmedik bir formatta
    yanıt verdiği durumlarda pipeline'ın durmasını önler.

    Args:
        tracks: ``{track_id: TrackedObject}`` biçiminde takip durumu.
        frames: Kırpıntıların alınacağı kare listesi.
        backend: ``generate(images, prompt, max_tokens)`` arayüzüne sahip
            bir VLM backend'i (bkz. `src/models/vlm_backend.py`).
        config: ``enabled``, ``min_confidence``, ``max_vehicles``,
            ``padding_ratio``, ``max_tokens`` alanlarına sahip yapılandırma
            nesnesi (`VehicleLabelingConfig`).

    Returns:
        ``{track_id: {"vehicle_type", "confidence_hint", "reasoning"}}``
        biçiminde etiket haritası. Yapılandırma kapalıysa, aday yoksa veya
        herhangi bir hata/parse başarısızlığı oluşursa boş sözlük.
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
    """İsimlendirme sonucunu, EventEngine'in okuduğu tüm veri yollarına işler.

    :func:`label_vehicles`'ın döndürdüğü harita, üç bağımsız veri
    temsilinde de güncellenmelidir; çünkü `src/events/event_engine.py`
    içindeki :class:`~src.events.event_engine.EventEngine` bu üç yolu da
    okur ve tutarsız bir güncelleme (örn. sadece track nesnelerini
    güncelleyip observation dict'lerini atlamak), kural motorunun eski
    ``"arac"`` etiketini görmesine yol açar:

      1. Canlı :class:`~src.perception.tracker.TrackedObject` nesneleri —
         ``class_name`` ve geçmişteki (``history``) ilgili
         :class:`~src.perception.detector.Detection` kayıtlarının
         ``class_name`` alanı.
      2. ``observation["detections"][*]["class"]`` — ``track_id`` üzerinden
         eşleştirilir.
      3. ``observation["tracks"][*]["class"]`` — ``track_id`` üzerinden
         eşleştirilir.
      4. ``observation["scene_graph"]["nodes"][*]["class"]`` **ve**
         ``"node_id"`` — düğüm kimliği
         :meth:`src.perception.scene_graph.SceneGraph.from_detections`
         tarafından ``"{class}_{track_id}"`` biçiminde üretildiğinden,
         sınıf değiştiğinde kimliğin de yeniden üretilmesi gerekir; aksi
         halde düğüm kimliği ile gerçek sınıf adı arasında tutarsızlık
         oluşur.

    Args:
        tracks: ``{track_id: TrackedObject}`` biçiminde canlı takip durumu.
        observations: :meth:`src.perception.observer_agent.ObserverAgent.observe_video`
            tarafından üretilmiş, halihazırda oluşturulmuş gözlem sözlükleri
            listesi. Bu liste **yerinde** (in-place) değiştirilir.
        label_map: :func:`label_vehicles` çıktısı.

    Returns:
        Etiketlenen (``label_map``'te bulunan) benzersiz track sayısı.
        ``label_map`` boşsa ``0`` ve hiçbir veri değiştirilmez.
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
