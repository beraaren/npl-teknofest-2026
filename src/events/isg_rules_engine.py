"""
İş Sağlığı ve Güvenliği (İSG) Kurallar Motoru ve Olay Analiz Sistemi.

Bu modül, ByteTrack ile tespit ve takibi yapılan nesneleri (Personel, Baret, Yelek, Forklift,
İş Makinesi, Yangın, Duman vb.) kare kare analiz ederek İSG kurallarına, ISO 45001 / 6331 sayılı
İSG Kanunu standartlarına göre risk seviyeli denetim, mekansal komşuluk, dinamik tehlike bölgesi
(Danger Zone) ve olay süreklilik raporları üretir.

Kural Eşikleri:
  - KKD Çakışma Eşiği (IoU/Overlap): >= 0.3
  - Baret Baş Bölgesi: İnsan yüksekliğinin üst %45'lik alanı
  - Yelek Gövde Bölgesi: İnsan yüksekliğinin %15 - %85'lik alanı
  - Forklift Devrilme (Yan Yatma) En/Boy Oranı: > 1.6
  - Forklift / Makine Hız Sınırı: 15.0 km/h
  - Güvenli Çalışma Mesafesi (İş Makinesi - Yaya): 3.0 m
  - Danger Zone Kırmızı Bölge (Kritik): < 2.0 m
  - Danger Zone Sarı Bölge (Yaklaşım): < 4.0 m (Hızlı araçlarda 5.0 m)
  - Yangın/Duman Etki Alanı: < 5.0 m
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd


# =====================================================================
# SINIF NORMALİZASYONU VE GERÇEK BOYUT METRİK REFERANSLARI
# =====================================================================

class CanonicalClass(str, Enum):
    """Sistem genelinde standartlaştırılmış kanonik sınıf tanımları."""
    PERSON = "person"
    HELMET = "helmet"
    VEST = "vest"
    VEHICLE = "arac"
    FIRE = "fire"
    SMOKE = "smoke"
    UNKNOWN = "unknown"

    @classmethod
    def normalize(cls, name: str) -> str:
        """Herhangi bir Türkçe/İngilizce sınıf ismini standart kanonik ada dönüştürür."""
        if not name:
            return cls.UNKNOWN.value
        norm = str(name).lower().strip()
        return CANONICAL_CLASS_MAP.get(norm, norm)


CANONICAL_CLASS_MAP: Dict[str, str] = {
    # İnsan / Personel
    "person": CanonicalClass.PERSON.value,
    "insan": CanonicalClass.PERSON.value,
    "personel": CanonicalClass.PERSON.value,
    "isci": CanonicalClass.PERSON.value,
    "işçi": CanonicalClass.PERSON.value,
    "human": CanonicalClass.PERSON.value,
    "worker": CanonicalClass.PERSON.value,
    # Baret / Kask
    "helmet": CanonicalClass.HELMET.value,
    "baret": CanonicalClass.HELMET.value,
    "kask": CanonicalClass.HELMET.value,
    "hard_hat": CanonicalClass.HELMET.value,
    "hardhat": CanonicalClass.HELMET.value,
    # Yelek / İkaz Yeleği
    "vest": CanonicalClass.VEST.value,
    "yelek": CanonicalClass.VEST.value,
    "ikaz_yelegi": CanonicalClass.VEST.value,
    "ikaz_yeleği": CanonicalClass.VEST.value,
    "safety_vest": CanonicalClass.VEST.value,
    # Araçlar (Forklift + Car + Machinery + Truck -> HEPSİ TEK SINIF)
    "arac": CanonicalClass.VEHICLE.value,
    "araç": CanonicalClass.VEHICLE.value,
    "forklift": CanonicalClass.VEHICLE.value,
    "machinery": CanonicalClass.VEHICLE.value,
    "is_makinesi": CanonicalClass.VEHICLE.value,
    "iş_makinesi": CanonicalClass.VEHICLE.value,
    "makine": CanonicalClass.VEHICLE.value,
    "machine": CanonicalClass.VEHICLE.value,
    "car": CanonicalClass.VEHICLE.value,
    "araba": CanonicalClass.VEHICLE.value,
    "truck": CanonicalClass.VEHICLE.value,
    "kamyon": CanonicalClass.VEHICLE.value,
    # Yangın / Ateş
    "fire": CanonicalClass.FIRE.value,
    "yangin": CanonicalClass.FIRE.value,
    "yangın": CanonicalClass.FIRE.value,
    "ates": CanonicalClass.FIRE.value,
    "ateş": CanonicalClass.FIRE.value,
    "flame": CanonicalClass.FIRE.value,
    # Duman
    "smoke": CanonicalClass.SMOKE.value,
    "duman": CanonicalClass.SMOKE.value,
}

CLASS_DISPLAY_NAMES_TR: Dict[str, str] = {
    CanonicalClass.PERSON.value: "Personel",
    CanonicalClass.HELMET.value: "Baret",
    CanonicalClass.VEST.value: "Yelek",
    CanonicalClass.VEHICLE.value: "Araç",
    CanonicalClass.FIRE.value: "Ateş/Yangın",
    CanonicalClass.SMOKE.value: "Duman",
}


class RiskSeverity(int, Enum):
    """ISO 45001 / 6331 sayılı İSG Kanunu Uyumlu Risk Derecelendirmesi"""
    INFO = 1       # Bilgilendirme / Güvenli
    LOW = 2        # Düşük Risk (Sarı Danger Zone Yaklaşımı)
    MEDIUM = 3     # Orta/Yüksek Risk (Tekli KKD Eksikliği, Hız Aşımı)
    HIGH = 4       # Yüksek/Kritik Risk (Tam KKD Yokluğu, Yakınlık İhlali, Duman)
    CRITICAL = 5   # Çok Yüksek / Acil Durum (Yangın, Devrilme, Kırmızı Danger Zone)


# =====================================================================
# TİP GÜVENLİĞİ YÜKSEK VERİ YAPILARI (DATACLASS)
# =====================================================================

@dataclass
class BoundingBox:
    """Güvenli koordinat, alan, merkez ve en/boy hesaplayıcıları."""
    x1: float
    y1: float
    x2: float
    y2: float
    width: float = field(init=False)
    height: float = field(init=False)
    area: float = field(init=False)
    center_x: float = field(init=False)
    center_y: float = field(init=False)

    def __post_init__(self):
        # Koordinat sıralama güvenliği (x1 <= x2, y1 <= y2)
        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1
        if self.y1 > self.y2:
            self.y1, self.y2 = self.y2, self.y1

        self.width = max(0.0, float(self.x2 - self.x1))
        self.height = max(0.0, float(self.y2 - self.y1))
        self.area = float(self.width * self.height)
        self.center_x = float((self.x1 + self.x2) / 2.0)
        self.center_y = float((self.y1 + self.y2) / 2.0)

    @property
    def aspect_ratio(self) -> float:
        """En / Boy (width / height) oranı."""
        return self.width / self.height if self.height > 0 else 1.0

    @property
    def safe_xyxy(self) -> Tuple[float, float, float, float]:
        """Güvenli (x1, y1, x2, y2) tuple'ı."""
        return (self.x1, self.y1, self.x2, self.y2)

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
        return cls(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BoundingBox:
        x1 = float(d.get("x1", 0))
        y1 = float(d.get("y1", 0))
        x2 = float(d.get("x2", x1 + float(d.get("width", 0))))
        y2 = float(d.get("y2", y1 + float(d.get("height", 0))))
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)

    def to_dict(self) -> Dict[str, float]:
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "area": self.area,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "aspect_ratio": self.aspect_ratio,
        }


@dataclass
class DetectionRecord:
    """Tekil nesne tespit kaydı ve geometrik özellikleri."""
    image_name: str
    class_name: str
    track_id: Union[int, str]
    bbox: BoundingBox
    confidence: Optional[float] = None
    frame_no: Optional[int] = None
    raw_class: str = ""

    # Geriye dönük uyumluluk ve doğrudan alan erişim property'leri
    @property
    def x1(self) -> float:
        return self.bbox.x1

    @property
    def y1(self) -> float:
        return self.bbox.y1

    @property
    def x2(self) -> float:
        return self.bbox.x2

    @property
    def y2(self) -> float:
        return self.bbox.y2

    @property
    def width(self) -> float:
        return self.bbox.width

    @property
    def height(self) -> float:
        return self.bbox.height

    @property
    def area(self) -> float:
        return self.bbox.area

    @property
    def center_x(self) -> float:
        return self.bbox.center_x

    @property
    def center_y(self) -> float:
        return self.bbox.center_y

    def __post_init__(self):
        # Sınıf adı normalizasyonu
        if not self.raw_class:
            self.raw_class = str(self.class_name)
        self.class_name = CanonicalClass.normalize(self.raw_class)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], frame_no: Optional[int] = None) -> DetectionRecord:
        """Girdi sözlüğünden (dict) güvenli DetectionRecord nesnesi oluşturur."""
        bbox = BoundingBox.from_dict(d)
        conf = d.get("confidence")
        conf_val = float(conf) if conf is not None and not pd.isna(conf) else None
        raw_cls = str(d.get("class_name", "unknown"))
        f_no = frame_no if frame_no is not None else d.get("frame_no")

        return cls(
            image_name=str(d.get("image_name", "")),
            class_name=raw_cls,
            track_id=d.get("track_id", "?"),
            bbox=bbox,
            confidence=conf_val,
            frame_no=f_no,
            raw_class=raw_cls,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Sözlük (dict) formatına serileştirir."""
        return {
            "image_name": self.image_name,
            "class_name": self.class_name,
            "track_id": self.track_id,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "width": self.width,
            "height": self.height,
            "area": self.area,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "confidence": self.confidence,
            "frame_no": self.frame_no,
            "raw_class": self.raw_class,
        }

    def __getitem__(self, item: str) -> Any:
        """Eski kod ve testlerle sözlük (dict) geriye dönük uyumluluğu."""
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        """Sözlük get metodu uyumluluğu."""
        return getattr(self, item, default)


@dataclass
class ISGAlert:
    """İSG Kural İhlali ve Güvenlik Uyarısı Veri Modeli."""
    severity: RiskSeverity
    category: str
    affected_track_ids: List[Union[int, str]]
    detail: str
    action: str
    time_code: str = ""
    frame_idx: Optional[int] = None
    image_name: str = ""

    def formatted_message(self) -> str:
        """Konsol ve raporlar için standart formatlanmış uyarı metnini üretir."""
        return f"{self.detail}\n     └─ 💡 Önerilen Aksiyon: {self.action}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "severity_name": self.severity.name,
            "category": self.category,
            "affected_track_ids": self.affected_track_ids,
            "detail": self.detail,
            "action": self.action,
            "time_code": self.time_code,
            "frame_idx": self.frame_idx,
            "image_name": self.image_name,
        }


@dataclass
class ISGReportSummary:
    """Kapsamlı İSG Denetim İstatistik ve Metrik Veri Yapısı."""
    total_images: int = 0
    total_detections: int = 0
    total_warnings: int = 0
    kkd_violations: int = 0
    tipping_alerts: int = 0
    proximity_alerts: int = 0
    fire_smoke_alerts: int = 0
    total_neighbors: int = 0
    danger_zone_red_breaches: int = 0
    danger_zone_yellow_alerts: int = 0

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def __setitem__(self, item: str, value: Any) -> None:
        setattr(self, item, value)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


# =====================================================================
# NESNE TAKİP VE HAREKET GEÇMİŞİ HAFIZASI
# =====================================================================

class ISGTrackingStore:
    """
    Kareler arası nesne takibi (track_id), zaman ve hareket geçmişi tutan hafıza sınıfı.
    Takip kopmalarında (uzun kare boşluklarında) sahte hız sıçramalarını önler.
    """
    def __init__(self):
        self.history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.active_objects: Dict[str, Dict[str, Any]] = {}

    def update_object(self, obj_id: str, record: Union[dict, DetectionRecord], frame_no: Optional[int] = None):
        if isinstance(record, DetectionRecord):
            rec_dict = record.to_dict()
        else:
            rec_dict = dict(record)

        if frame_no is not None:
            rec_dict["frame_no"] = frame_no
        elif "frame_no" not in rec_dict:
            rec_dict["frame_no"] = None

        self.history[obj_id].append(rec_dict)
        self.active_objects[obj_id] = rec_dict

    def get_previous_record(self, obj_id: str) -> Optional[Dict[str, Any]]:
        """Geriye dönük uyumlu: İlgili nesnenin bir önceki kaydını döner."""
        if len(self.history[obj_id]) >= 2:
            return self.history[obj_id][-2]
        return None

    def get_valid_previous_record(
        self,
        obj_id: str,
        current_frame: Optional[int],
        max_frame_gap: int = 10,
    ) -> Optional[Tuple[Dict[str, Any], int]]:
        """
        Geçerli bir önceki kaydı ve onun gerçek kare indeksini döner.
        Eğer nesne çok uzun süre kadrajda görünmemişse (max_frame_gap aşılmışsa) None döner.
        """
        if len(self.history[obj_id]) < 2:
            return None

        prev_rec = self.history[obj_id][-2]
        prev_frame = prev_rec.get("frame_no")

        if current_frame is not None and prev_frame is not None:
            if current_frame - prev_frame > max_frame_gap:
                return None
            return prev_rec, prev_frame

        return prev_rec, (current_frame - 1 if current_frame else 0)


# =====================================================================
# İSG KURALLAR MOTORU (ISGRulesEngine)
# =====================================================================

class ISGRulesEngine:
    """
    İş Sağlığı ve Güvenliği (İSG) Kurallar Motoru.
    Geliştirmeye açık modüler mimarisiyle İSG senaryolarını, kural ihlallerini,
    mekansal komşulukları ve tehlike bölgelerini değerlendirir.
    """

    # Sınıf bazlı gerçek boyut referansları (Metre) — mesafe ve hız ölçeklemesi için
    CLASS_HEIGHTS_M = {
        CanonicalClass.PERSON.value: 1.70,
        CanonicalClass.VEHICLE.value: 2.00,
    }
    DEFAULT_HEIGHT_M = 1.70

    def __init__(self, fps: float = 30.0):
        self.fps = max(1.0, float(fps))

    @staticmethod
    def _normalize_box(box: Union[dict, DetectionRecord]) -> Dict[str, Any]:
        """Girdi sözlük veya DetectionRecord ise standart anahtarlara sahip dict formatına getirir."""
        if isinstance(box, DetectionRecord):
            return box.to_dict()

        x1 = float(box.get("x1", 0))
        y1 = float(box.get("y1", 0))
        x2 = float(box.get("x2", 0))
        y2 = float(box.get("y2", 0))
        width = float(box.get("width", x2 - x1))
        height = float(box.get("height", y2 - y1))
        area = float(box.get("area", width * height))
        center_x = float(box.get("center_x", (x1 + x2) / 2.0))
        center_y = float(box.get("center_y", (y1 + y2) / 2.0))

        cls_name = str(box.get("class_name", "unknown"))
        cls_name = CanonicalClass.normalize(cls_name)

        return {
            **box,
            "class_name": cls_name,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": width,
            "height": height,
            "area": area,
            "center_x": center_x,
            "center_y": center_y,
            "track_id": box.get("track_id", "?"),
        }

    @staticmethod
    def is_inside_or_overlapping(boxA: Union[dict, DetectionRecord], boxB: Union[dict, DetectionRecord], threshold: float = 0.3) -> bool:
        """
        boxB'nin (ör. baret/yelek) boxA (ör. insan) ile çakışıp çakışmadığını kontrol eder.
        Çakışma oranı: Kesişim Alanı / boxB Alanı >= threshold
        """
        xA = max(boxA["x1"], boxB["x1"])
        yA = max(boxA["y1"], boxB["y1"])
        xB = min(boxA["x2"], boxB["x2"])
        yB = min(boxA["y2"], boxB["y2"])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxBArea = boxB["area"] if boxB["area"] > 0 else (boxB["x2"] - boxB["x1"]) * (boxB["y2"] - boxB["y1"])

        if boxBArea <= 0:
            return False

        overlap = interArea / float(boxBArea)
        return overlap >= threshold

    def _meters_per_pixel(self, box1: Union[dict, DetectionRecord], box2: Union[dict, DetectionRecord]) -> float:
        """
        Ölçeği sınıfı bilinen nesnenin kendi gerçek boyutundan kurar.
        İnsan varsa insan (1.70m), yoksa forklift (2.00m) veya machinery (2.50m) referans alınır.
        """
        b1 = self._normalize_box(box1)
        b2 = self._normalize_box(box2)

        for box in (b1, b2):
            if box.get("class_name") == CanonicalClass.PERSON.value and box["height"] > 0:
                return self.CLASS_HEIGHTS_M[CanonicalClass.PERSON.value] / float(box["height"])

        for box in (b1, b2):
            h_m = self.CLASS_HEIGHTS_M.get(box.get("class_name"))
            if h_m and box["height"] > 0:
                return h_m / float(box["height"])

        avg_pixel_height = (b1["height"] + b2["height"]) / 2.0
        if avg_pixel_height <= 0:
            return 0.0
        return self.DEFAULT_HEIGHT_M / avg_pixel_height

    def estimate_real_distance(self, box1: Union[dict, DetectionRecord], box2: Union[dict, DetectionRecord]) -> float:
        """
        İki nesne arasındaki 2D merkez mesafesini metre cinsinden kestirir.
        """
        b1 = self._normalize_box(box1)
        b2 = self._normalize_box(box2)

        meters_per_pixel = self._meters_per_pixel(b1, b2)
        if meters_per_pixel == 0:
            return 999.0
        pixel_dist = math.hypot(b1["center_x"] - b2["center_x"], b1["center_y"] - b2["center_y"])
        return pixel_dist * meters_per_pixel

    @staticmethod
    def extract_frame_number(image_name: str) -> Optional[int]:
        """
        Görüntü dosya adından kare numarasını (frame index) çıkarmaya çalışır.
        """
        match = re.search(r"(?:frame_|Frame_No|img_|image_)(\d+)", image_name, re.IGNORECASE)
        if match:
            return int(match.group(1))
        # Başka bir desende tekil sayı yakalama
        num_match = re.search(r"\b(\d+)\b", image_name)
        if num_match:
            try:
                return int(num_match.group(1))
            except ValueError:
                pass
        return None

    def estimate_speed_kmh(
        self,
        curr_box: Union[dict, DetectionRecord],
        prev_box: Optional[Union[dict, DetectionRecord]],
        curr_frame: Optional[int] = None,
        prev_frame: Optional[int] = None,
    ) -> float:
        """
        İki kare arasındaki piksel kaymasını kare farkı ve zaman değişimine (dt)
        bölerek km/saat cinsinden hesaplar.
        """
        if prev_box is None:
            return 0.0

        b_curr = self._normalize_box(curr_box)
        b_prev = self._normalize_box(prev_box)

        if b_curr["height"] <= 0:
            return 0.0

        if curr_frame is not None and prev_frame is not None and curr_frame > prev_frame:
            time_delta = (curr_frame - prev_frame) / float(self.fps)
        else:
            time_delta = 1.0 / float(self.fps)

        if time_delta <= 0:
            return 0.0

        meters_per_pixel = self._meters_per_pixel(b_curr, b_prev)
        if meters_per_pixel == 0:
            return 0.0

        pixel_disp = math.hypot(b_curr["center_x"] - b_prev["center_x"], b_curr["center_y"] - b_prev["center_y"])
        distance_m = pixel_disp * meters_per_pixel

        speed_m_per_sec = distance_m / time_delta
        speed_kmh = speed_m_per_sec * 3.6
        return round(speed_kmh, 1)

    @staticmethod
    def _get_relative_position(box1: Union[dict, DetectionRecord], box2: Union[dict, DetectionRecord]) -> str:
        """
        box1'in box2'ye göre göreceli mekansal konumunu (Sol, Sağ, Ön, Arka) döndürür.
        """
        dx = box1["center_x"] - box2["center_x"]
        dy = box1["center_y"] - box2["center_y"]

        horiz = "Sağ" if dx > 30 else ("Sol" if dx < -30 else "")
        vert = "Arka" if dy > 30 else ("Ön" if dy < -30 else "")

        if horiz and vert:
            return f"{horiz}-{vert}"
        elif horiz:
            return horiz
        elif vert:
            return vert
        return "Yakın Çevre"

    # -------------------------------------------------------------
    # KKD DOĞRULAMA (Association & Geometrik Alanlar)
    # -------------------------------------------------------------
    def evaluate_person_ppe(
        self,
        person: Union[dict, DetectionRecord],
        equipment_boxes: List[Union[dict, DetectionRecord]],
        used_equipment_ids: Optional[Set[Any]] = None,
    ) -> Tuple[bool, bool]:
        """
        Personele ait Baret ve Yelek kullanım durumunu doğrular.
        - Baret: Baş bölgesi (Üst %45 -> center_y <= y1 + 0.45 * height)
        - Yelek: Gövde bölgesi (%15 - %85 -> y1 + 0.15 * height <= center_y <= y1 + 0.85 * height)
        - used_equipment_ids: Greedy 1-to-1 eşleşme ile tek ekipmanın birden fazla kişiye mükerrer atanmasını önler.
        """
        p = self._normalize_box(person)
        has_helmet = False
        has_vest = False

        head_boundary_y = p["y1"] + (p["height"] * 0.45)
        body_top_y = p["y1"] + (p["height"] * 0.15)
        body_bottom_y = p["y1"] + (p["height"] * 0.85)

        for eq_raw in equipment_boxes:
            eq = self._normalize_box(eq_raw)
            eq_id = eq.get("track_id")

            # Eğer greedy 1-to-1 eşleşme isteniyorsa kullanılmış ekipmanı atla
            if used_equipment_ids is not None and eq_id in used_equipment_ids:
                continue

            if self.is_inside_or_overlapping(p, eq, threshold=0.3):
                cls = eq["class_name"]
                if cls == CanonicalClass.HELMET.value and not has_helmet:
                    if eq["center_y"] <= head_boundary_y:
                        has_helmet = True
                        if used_equipment_ids is not None:
                            used_equipment_ids.add(eq_id)
                elif cls == CanonicalClass.VEST.value and not has_vest:
                    if body_top_y <= eq["center_y"] <= body_bottom_y:
                        has_vest = True
                        if used_equipment_ids is not None:
                            used_equipment_ids.add(eq_id)

        return has_helmet, has_vest

    # -------------------------------------------------------------
    # İSG KURAL 1: KKD (Baret & Yelek) Kontrolü
    # -------------------------------------------------------------
    def check_kkd(
        self,
        person: Union[dict, DetectionRecord],
        equipment_boxes: List[Union[dict, DetectionRecord]],
        used_equipment_ids: Optional[Set[Any]] = None,
    ) -> List[str]:
        warnings = []
        p = self._normalize_box(person)
        has_helmet, has_vest = self.evaluate_person_ppe(p, equipment_boxes, used_equipment_ids=used_equipment_ids)
        person_id = f"Personel-{p['track_id']}"

        if not has_helmet and not has_vest:
            warnings.append(
                f"[KRİTİK İHLAL | RİSK SEVİYESİ 4] ⚠️ {person_id}: Baret ve Yelek TAKMIYOR! (Eksik KKD Kullanımı)\n"
                f"     └─ 💡 Önerilen Aksiyon: Personele KKD ekipmanları derhal temin edilmeli ve sahaya baret/yeleksiz girişi engellenmelidir."
            )
        elif not has_helmet:
            warnings.append(
                f"[YÜKSEK İHLAL | RİSK SEVİYESİ 3] ⚠️ {person_id}: Baret TAKMIYOR! (Baş Koruyucu Eksik)\n"
                f"     └─ 💡 Önerilen Aksiyon: Personele baret verilmeli ve kullanımı denetlenmelidir."
            )
        elif not has_vest:
            warnings.append(
                f"[YÜKSEK İHLAL | RİSK SEVİYESİ 3] ⚠️ {person_id}: Koruyucu Yelek TAKMIYOR! (İkaz Yeleği Eksik)\n"
                f"     └─ 💡 Önerilen Aksiyon: Yüksek görünürlüklü reflektörlü ikaz yeleği temin edilmelidir."
            )

        return warnings

    # -------------------------------------------------------------
    # İSG KURAL 2: Forklift Yan Yatması / Devrilme & Hız Kontrolü
    # -------------------------------------------------------------
    def check_forklift_status(
        self,
        forklift: Union[dict, DetectionRecord],
        prev_forklift: Optional[Union[dict, DetectionRecord]] = None,
        curr_frame: Optional[int] = None,
        prev_frame: Optional[int] = None,
    ) -> List[str]:
        warnings = []
        f = self._normalize_box(forklift)
        forklift_id = f"Forklift-{f['track_id']}"

        aspect_ratio = f["width"] / float(f["height"]) if f["height"] > 0 else 1.0

        # Normal forklift dik durduğunda Genişlik / Yükseklik oranı düşüktür.
        # Yan yattığında genişlik belirgin şekilde yükseklikten fazla olur (Eşik: > 1.6).
        if aspect_ratio > 1.6:
            warnings.append(
                f"[ACİL DURUM | RİSK SEVİYESİ 5] 🚨 KRİTİK ALARM: {forklift_id} Yan Yatmış / Devrilmiş Olabilir! (En/Boy Oranı: {aspect_ratio:.2f} > 1.6)\n"
                f"     └─ 💡 Önerilen Aksiyon: Saha sorumlusu ve acil müdahale ekibi derhal bilgilendirilmeli, araç çevresi emniyete alınmalıdır."
            )

        # Hız kontrolü
        if prev_forklift is not None:
            speed = self.estimate_speed_kmh(f, prev_forklift, curr_frame=curr_frame, prev_frame=prev_frame)
            if speed > 15.0:
                warnings.append(
                    f"[HIZ İHLALİ | RİSK SEVİYESİ 3] ⚠️ {forklift_id}: Depo içi aşırı hız yapıyor! (Tahmini Hız: {speed} km/h > Sınır: 15.0 km/h)\n"
                    f"     └─ 💡 Önerilen Aksiyon: Forklift operatörüne hız sınırı uyarısı iletilmeli, hız sınırlayıcı donanım denetlenmelidir."
                )

        return warnings

    # -------------------------------------------------------------
    # İSG KURAL 3: Tehlikeli Yakınlık Kontrolü (İnsan - İş Makinesi)
    # -------------------------------------------------------------
    def check_proximity(
        self,
        persons: List[Union[dict, DetectionRecord]],
        vehicles: List[Union[dict, DetectionRecord]],
        min_safe_meters: float = 3.0,
    ) -> List[str]:
        warnings = []
        for p_raw in persons:
            p = self._normalize_box(p_raw)
            p_id = f"Personel-{p['track_id']}"
            for v_raw in vehicles:
                v = self._normalize_box(v_raw)
                v_display = CLASS_DISPLAY_NAMES_TR.get(v["class_name"], v["class_name"].capitalize())
                v_id = f"{v_display}-{v['track_id']}"
                dist_m = self.estimate_real_distance(p, v)

                if dist_m < min_safe_meters:
                    rel_pos = self._get_relative_position(p, v)
                    warnings.append(
                        f"[KRİTİK TEHLİKE | RİSK SEVİYESİ 4] 🚨 KRİTİK TEHLİKE: {p_id} ile {v_id} tehlikeli derecede yakın! (Mesafe: ~{dist_m:.1f}m < 3.0m, Konum: {rel_pos})\n"
                        f"     └─ 💡 Önerilen Aksiyon: İş makinesi çalışma alanında yaya bulunmamalı; araç durdurulup yaya emniyetli bölgeye çekilmelidir."
                    )
        return warnings

    # -------------------------------------------------------------
    # İSG KURAL 4: Yangın ve Duman Kontrolü
    # -------------------------------------------------------------
    def check_fire_and_smoke(self, hazards: List[Union[dict, DetectionRecord]]) -> List[str]:
        warnings = []
        for h_raw in hazards:
            h = self._normalize_box(h_raw)
            h_id = f"{h['class_name'].capitalize()}-{h['track_id']}"
            if h["class_name"] == CanonicalClass.FIRE.value:
                warnings.append(
                    f"[ACİL DURUM | RİSK SEVİYESİ 5] 🔥 ACİL DURUM: {h_id} (Ateş/Yangın) algılandı! Derhal müdahale edilmeli!\n"
                    f"     └─ 💡 Önerilen Aksiyon: Yangın alarmı aktive edilmeli, söndürme ekibi yönlendirilmeli ve saha tahliye edilmelidir."
                )
            elif h["class_name"] == CanonicalClass.SMOKE.value:
                warnings.append(
                    f"[YÜKSEK UYARI | RİSK SEVİYESİ 4] 💨 ACİL UYARI: {h_id} (Duman) algılandı! Yangın riski kontrol edilmeli.\n"
                    f"     └─ 💡 Önerilen Aksiyon: Duman kaynağı incelenmeli, havalandırma kapakları ve yangın sensörleri kontrol edilmelidir."
                )
        return warnings

    # -------------------------------------------------------------
    # İSG KURAL 5: Birbirine Yakın Komşular Analizi (Mekansal Proximity)
    # -------------------------------------------------------------
    def analyze_neighbors(
        self,
        persons: List[Union[dict, DetectionRecord]],
        vehicles: List[Union[dict, DetectionRecord]],
        equipments: List[Union[dict, DetectionRecord]],
        person_person_threshold: float = 1.5,
        vehicle_vehicle_threshold: float = 4.0,
        unattached_eq_threshold: float = 2.0,
    ) -> Dict[str, List[str]]:
        """
        Görüntüdeki varlıklar arası mekansal yakınlık/komşuluk ilişkilerini analiz eder.
        """
        results: Dict[str, List[str]] = {
            "person_person": [],
            "vehicle_vehicle": [],
            "unattached_equipment": [],
        }

        norm_persons = [self._normalize_box(p) for p in persons]
        norm_vehicles = [self._normalize_box(v) for v in vehicles]
        norm_equipments = [self._normalize_box(e) for e in equipments]

        # 1. Personel - Personel Komşuluğu (< 1.5m)
        for i in range(len(norm_persons)):
            for j in range(i + 1, len(norm_persons)):
                p1, p2 = norm_persons[i], norm_persons[j]
                dist_m = self.estimate_real_distance(p1, p2)
                if dist_m <= person_person_threshold:
                    rel_pos = self._get_relative_position(p2, p1)
                    results["person_person"].append(
                        f"👥 Personel-{p1['track_id']} ile Personel-{p2['track_id']} yakın komşu (Mesafe: {dist_m:.1f}m - Konum: {rel_pos})"
                    )

        # 2. Araç - Araç Komşuluğu (< 4.0m)
        for i in range(len(norm_vehicles)):
            for j in range(i + 1, len(norm_vehicles)):
                v1, v2 = norm_vehicles[i], norm_vehicles[j]
                dist_m = self.estimate_real_distance(v1, v2)
                if dist_m <= vehicle_vehicle_threshold:
                    v1_name = CLASS_DISPLAY_NAMES_TR.get(v1["class_name"], v1["class_name"].capitalize())
                    v2_name = CLASS_DISPLAY_NAMES_TR.get(v2["class_name"], v2["class_name"].capitalize())
                    v1_id = f"{v1_name}-{v1['track_id']}"
                    v2_id = f"{v2_name}-{v2['track_id']}"
                    rel_pos = self._get_relative_position(v2, v1)
                    results["vehicle_vehicle"].append(
                        f"⚠️ Araç Yakınlaşması: {v1_id} ile {v2_id} yakın alanda çalışıyor (Mesafe: {dist_m:.1f}m - Konum: {rel_pos})"
                    )

        # 3. Sahipsiz / Serbest KKD Komşuluğu (< 2.0m)
        for eq in norm_equipments:
            attached = any(self.is_inside_or_overlapping(p, eq) for p in norm_persons)
            if not attached:
                for p in norm_persons:
                    dist_m = self.estimate_real_distance(p, eq)
                    if dist_m <= unattached_eq_threshold:
                        eq_name = "Baret" if eq["class_name"] == CanonicalClass.HELMET.value else "Yelek"
                        rel_pos = self._get_relative_position(eq, p)
                        results["unattached_equipment"].append(
                            f"🪖 Sahipsiz/Serbest Ekipman: {eq_name}-{eq['track_id']}, Personel-{p['track_id']} yakınında (Mesafe: {dist_m:.1f}m - Konum: {rel_pos})"
                        )

        return results

    # -------------------------------------------------------------
    # İSG KURAL 6: Danger Zone (Tehlike Bölgesi) Analizi
    # -------------------------------------------------------------
    def analyze_danger_zones(
        self,
        persons: List[Union[dict, DetectionRecord]],
        vehicles: List[Union[dict, DetectionRecord]],
        hazards: List[Union[dict, DetectionRecord]],
        store: Optional[ISGTrackingStore] = None,
        curr_frame: Optional[int] = None,
    ) -> List[str]:
        """
        Dinamik Tehlike Bölgesi (Danger Zone) ihlallerini ve yaklaşımlarını analiz eder.
        - Kırmızı Bölge (Kritik: < 2.0m)
        - Sarı Bölge (Uyarı/Yaklaşım: 2.0m - 4.0m / hızlı araçlarda 5.0m)
        - Tehlike Bölgesi (Yangın/Duman: < 5.0m)
        """
        danger_alerts = []
        norm_persons = [self._normalize_box(p) for p in persons]
        norm_vehicles = [self._normalize_box(v) for v in vehicles]
        norm_hazards = [self._normalize_box(h) for h in hazards]

        # 1. Araç Bazlı Danger Zone'lar
        for v in norm_vehicles:
            v_display = CLASS_DISPLAY_NAMES_TR.get(v["class_name"], v["class_name"].capitalize())
            v_id = f"{v_display}-{v['track_id']}"
            speed = 0.0
            if store and v["class_name"] == CanonicalClass.VEHICLE.value:
                prev_v = store.get_previous_record(f"forklift_{v['track_id']}")
                if prev_v:
                    speed = self.estimate_speed_kmh(
                        v, prev_v, curr_frame=curr_frame, prev_frame=curr_frame - 1 if curr_frame else None
                    )

            yellow_radius = 5.0 if speed > 10.0 else 4.0

            for p in norm_persons:
                p_id = f"Personel-{p['track_id']}"
                dist_m = self.estimate_real_distance(p, v)

                if dist_m < 2.0:
                    rel_pos = self._get_relative_position(p, v)
                    danger_alerts.append(
                        f"[KRİTİK DANGER ZONE İHLALİ | RİSK SEVİYESİ 5] 🚨 KRİTİK DANGER ZONE İHLALİ: {p_id}, {v_id}'nin Kırmızı Bölgesinde (Konum: {rel_pos})! (Mesafe: {dist_m:.1f}m < 2.0m)"
                    )
                elif dist_m < yellow_radius:
                    rel_pos = self._get_relative_position(p, v)
                    speed_note = f" [Araç Hızlı: {speed} km/h]" if speed > 10.0 else ""
                    danger_alerts.append(
                        f"[DANGER ZONE UYARISI | RİSK SEVİYESİ 2] ⚠️ DANGER ZONE UYARISI: {p_id}, {v_id}'nin Sarı Yaklaşım Bölgesinde (Konum: {rel_pos})! (Mesafe: {dist_m:.1f}m){speed_note}"
                    )

        # 2. Yangın ve Duman Danger Zone'u
        for h in norm_hazards:
            h_id = f"{h['class_name'].capitalize()}-{h['track_id']}"
            h_name = "Yangın" if h["class_name"] == CanonicalClass.FIRE.value else "Duman"
            for p in norm_persons:
                p_id = f"Personel-{p['track_id']}"
                dist_m = self.estimate_real_distance(p, h)
                if dist_m < 5.0:
                    rel_pos = self._get_relative_position(p, h)
                    danger_alerts.append(
                        f"[TEHLİKE BÖLGESİ İHLALİ | RİSK SEVİYESİ 5] 🔥 TEHLİKE BÖLGESİ İHLALİ: {p_id}, {h_id} ({h_name}) etki alanında (Konum: {rel_pos})! (Mesafe: {dist_m:.1f}m < 5.0m)"
                    )

        return danger_alerts

    def describe_detection(
        self,
        record: Union[dict, DetectionRecord],
        equipments: Optional[List[Union[dict, DetectionRecord]]] = None,
        prev_record: Optional[Union[dict, DetectionRecord]] = None,
        curr_frame: Optional[int] = None,
        prev_frame: Optional[int] = None,
    ) -> str:
        """
        ByteTrack nesne tespit kaydını kesin geometrik özelliklerle Türkçe betimler.
        """
        r = self._normalize_box(record)
        cls = r.get("class_name", "unknown")
        track_id = r.get("track_id", "?")
        conf = r.get("confidence")
        conf_str = f"Güven: %{conf * 100:.1f} | " if conf is not None else ""

        x1, y1 = int(round(r.get("x1", 0))), int(round(r.get("y1", 0)))
        x2, y2 = int(round(r.get("x2", 0))), int(round(r.get("y2", 0)))
        w, h = int(round(r.get("width", x2 - x1))), int(round(r.get("height", y2 - y1)))
        cx, cy = r.get("center_x", (x1 + x2) / 2.0), r.get("center_y", (y1 + y2) / 2.0)

        bbox_str = f"Bbox: [{x1}, {y1}, {x2}, {y2}] (Merkez: ({cx:g}, {cy:g}), Boyut: {w}x{h}px)"

        if cls == CanonicalClass.PERSON.value:
            has_helmet, has_vest = self.evaluate_person_ppe(r, equipments or [])
            helmet_status = "VAR ✅" if has_helmet else "YOK ❌"
            vest_status = "VAR ✅" if has_vest else "YOK ❌"
            return f"👤 Personel-{track_id}: {conf_str}{bbox_str} | KKD -> Baret: {helmet_status}, Yelek: {vest_status}"

        elif cls == CanonicalClass.HELMET.value:
            return f"🪖 Baret-{track_id}: {conf_str}{bbox_str}"

        elif cls == CanonicalClass.VEST.value:
            return f"🦺 Yelek-{track_id}: {conf_str}{bbox_str}"

        elif cls == CanonicalClass.VEHICLE.value:
            aspect_ratio = w / float(h) if h > 0 else 1.0
            orient_str = f" | En/Boy: {aspect_ratio:.2f}"
            if aspect_ratio > 1.6:
                orient_str += " (Yan Yatmış / Devrilme Riski 🚨)"

            speed_str = ""
            if prev_record is not None:
                speed = self.estimate_speed_kmh(r, prev_record, curr_frame=curr_frame, prev_frame=prev_frame)
                speed_str = f" | Hız: {speed} km/h"

            return f"🚜 Araç-{track_id}: {conf_str}{bbox_str}{orient_str}{speed_str}"


        elif cls == CanonicalClass.FIRE.value:
            return f"🔥 Ateş/Yangın-{track_id}: {conf_str}{bbox_str} (Acil Durum Varlığı)"

        elif cls == CanonicalClass.SMOKE.value:
            return f"💨 Duman-{track_id}: {conf_str}{bbox_str} (Yangın Riski Belirtisi)"

        else:
            return f"📦 {cls.capitalize()}-{track_id}: {conf_str}{bbox_str}"


# =====================================================================
# ÇOK FORMATLI RAPOR ÜRETİCİLERİ
# =====================================================================

def _extract_semantic_key(warning_text: str) -> str:
    """
    Uyarı metninden dinamik sayısal değerler (mesafe, hız vb.) değişse bile
    süregelen aynı olayı tanıyan anlamsal olay anahtarı üretir.
    """
    first_line = warning_text.split("\n")[0]
    # Mesafe veya hız gibi değişken sayısal parçaları temizle
    clean_key = re.sub(r"\(Mesafe:[^)]+\)", "", first_line)
    clean_key = re.sub(r"\(Tahmini Hız:[^)]+\)", "", clean_key)
    clean_key = re.sub(r"\[Araç Hızlı:[^\]]+\]", "", clean_key)
    return clean_key.strip()


def generate_text_report(
    report_lines: List[str],
    summary_stats: Union[Dict[str, Any], ISGReportSummary],
    closed_streaks: List[Dict[str, Any]],
    fps: float = 30.0,
) -> str:
    """Standart konsol ve metin (.txt) formatında profesyonel İSG denetim raporu üretir."""
    lines = list(report_lines)
    stats_dict = summary_stats.to_dict() if isinstance(summary_stats, ISGReportSummary) else summary_stats

    lines.append("----------------------------------------------------------")
    lines.append("                  İSG RAPOR ÖZETİ                         ")
    lines.append("----------------------------------------------------------")
    lines.append(f"Analiz Edilen Toplam Görüntü Sayısı : {stats_dict['total_images']}")
    lines.append(f"Üretilen Toplam İSG Uyarısı Sayısı  : {stats_dict['total_warnings']}")
    lines.append(f"  • KKD (Baret/Yelek) İhlali         : {stats_dict['kkd_violations']}")
    lines.append(f"  • Forklift Yan Yatması / Devrilme  : {stats_dict['tipping_alerts']}")
    lines.append(f"  • Tehlikeli Yakınlık İhlali        : {stats_dict['proximity_alerts']}")
    lines.append(f"  • Yangın / Duman İhbarı            : {stats_dict['fire_smoke_alerts']}")
    lines.append(f"  • Mekansal Komşuluk Tespiti        : {stats_dict['total_neighbors']}")
    lines.append(f"  • Danger Zone Kırmızı Bölge İhlali : {stats_dict['danger_zone_red_breaches']}")
    lines.append(f"  • Danger Zone Sarı Bölge Uyarısı   : {stats_dict['danger_zone_yellow_alerts']}")

    if closed_streaks:
        lines.append("")
        lines.append("Uyarı Süreklilik Detayı (Kaç kare ve saniye sürdüğü):")
        for s in sorted(closed_streaks, key=lambda x: -x["frames"]):
            main_text = s["text"].split("\n")[0]
            dur_sec = s["frames"] / float(fps)
            frame_idx = ISGRulesEngine.extract_frame_number(s["image"])
            if frame_idx is not None:
                start_sec = frame_idx / float(fps)
                sm = int(start_sec // 60)
                ss = start_sec % 60
                time_str = f", zaman: {sm:02d}:{ss:04.1f}"
            else:
                time_str = ""
            lines.append(f"  • {main_text}  → {s['frames']} kare / {dur_sec:.1f} sn (ilk: {s['image']}{time_str})")

    lines.append("==========================================================")
    return "\n".join(lines)


def generate_json_report(
    summary_stats: Union[Dict[str, Any], ISGReportSummary],
    closed_streaks: List[Dict[str, Any]],
    json_frames: List[Dict[str, Any]],
    fps: float = 30.0,
    source_csv: str = "",
) -> Dict[str, Any]:
    """NLP, Karar Ajanları ve RAG sistemleri için makinece okunabilir JSON veri yapısı üretir."""
    stats_dict = summary_stats.to_dict() if isinstance(summary_stats, ISGReportSummary) else summary_stats
    return {
        "metadata": {
            "engine": "BERA ISGRulesEngine v2.0",
            "fps": fps,
            "source_csv": str(source_csv),
        },
        "summary_stats": stats_dict,
        "incident_streaks": [
            {
                "warning": s["text"].split("\n")[0],
                "action": s["text"].split("\n")[1].strip() if len(s["text"].split("\n")) > 1 else "",
                "duration_frames": s["frames"],
                "duration_seconds": round(s["frames"] / float(fps), 2),
                "first_image": s["image"],
            }
            for s in sorted(closed_streaks, key=lambda x: -x["frames"])
        ],
        "frames": json_frames,
    }


def generate_markdown_report(
    summary_stats: Union[Dict[str, Any], ISGReportSummary],
    closed_streaks: List[Dict[str, Any]],
    fps: float = 30.0,
) -> str:
    """Yönetici özeti ve görsel dashboard formatında Markdown raporu üretir."""
    stats_dict = summary_stats.to_dict() if isinstance(summary_stats, ISGReportSummary) else summary_stats
    md_lines = [
        "# 🛡️ BERA İSG (İş Sağlığı ve Güvenliği) Analiz Raporu",
        "",
        "> **Standart:** ISO 45001 & 6331 Sayılı İSG Kanunu Uyumlu Otomatik Saha Denetim Sistemi",
        "",
        "## 📊 Genel Saha Özeti",
        "",
        "| Metrik | Değer |",
        "| :--- | :--- |",
        f"| **Analiz Edilen Görüntü Sayısı** | `{stats_dict['total_images']}` |",
        f"| **Toplam Nesne Tespiti** | `{stats_dict['total_detections']}` |",
        f"| **Toplam İSG Uyarısı** | `{stats_dict['total_warnings']}` |",
        f"| **KKD (Baret/Yelek) İhlali** | `{stats_dict['kkd_violations']}` |",
        f"| **Forklift Yan Yatması / Devrilme** | `{stats_dict['tipping_alerts']}` |",
        f"| **Tehlikeli Yakınlık İhlali** | `{stats_dict['proximity_alerts']}` |",
        f"| **Yangın / Duman İhbarı** | `{stats_dict['fire_smoke_alerts']}` |",
        f"| **Danger Zone Kırmızı Bölge İhlali** | `{stats_dict['danger_zone_red_breaches']}` |",
        f"| **Danger Zone Sarı Bölge Uyarısı** | `{stats_dict['danger_zone_yellow_alerts']}` |",
        "",
        "## 🚨 Olay ve İhlal Süreklilik Detayları",
        "",
        "| Risk Seviyesi | Olay Açıklaması | Süre (Kare) | Süre (Sn) | İlk Tespit Edilen Kare |",
        "| :---: | :--- | :---: | :---: | :--- |",
    ]
    for s in sorted(closed_streaks, key=lambda x: -x["frames"]):
        txt = s["text"].split("\n")[0]
        risk = "CRITICAL 🔴" if "RİSK SEVİYESİ 5" in txt or "ACİL" in txt else (
            "HIGH 🟠" if "RİSK SEVİYESİ 4" in txt else "MEDIUM 🟡"
        )
        clean_t = txt.replace("|", "\\|")
        dur_s = s["frames"] / float(fps)
        md_lines.append(f"| {risk} | {clean_t} | {s['frames']} | {dur_s:.1f}s | `{s['image']}` |")

    return "\n".join(md_lines) + "\n"


# =====================================================================
# ANALİZ İCRA MOTORU (run_isg_analysis)
# =====================================================================

def run_isg_analysis(
    csv_path: str,
    output_txt_path: str,
    fps: float = 30.0,
    output_json_path: Optional[str] = None,
    output_md_path: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    ByteTrack CSV çıktısını okuyarak her kare için profesyonel Türkçe İSG raporları üretir.

    Desteklenen Çıktılar:
      1. Plain Text (.txt): Detaylı kare kare analiz ve süreklilik özeti.
      2. JSON (.json): Karar Ajanları, RAG ve VLM için yapılandırılmış veri.
      3. Markdown (.md): Yönetici özeti ve görsel risk dağılım dashboard'u.
    """
    df = pd.read_csv(csv_path)
    engine = ISGRulesEngine(fps=fps)
    store = ISGTrackingStore()

    grouped = df.groupby("image_name", sort=False)

    report_lines: List[str] = []
    json_frames: List[Dict[str, Any]] = []

    summary_stats = ISGReportSummary(
        total_images=len(grouped),
        total_detections=len(df),
    )

    # Streak takibi: semantic_key -> {"start": frame_no, "last": frame_no, "frames": n, "text": original_text, "image": image_name}
    active_streaks: Dict[str, Dict[str, Any]] = {}
    closed_streaks: List[Dict[str, Any]] = []
    frame_no = 0

    report_lines.append("==========================================================")
    report_lines.append("     GÖRÜNTÜ İŞLEME İSG (İŞ SAĞLIĞI VE GÜVENLİĞİ) RAPORU   ")
    report_lines.append("==========================================================\n")

    for image_name, frame_df in grouped:
        frame_no += 1
        raw_records = frame_df.to_dict("records")

        # DetectionRecord nesnelerini oluştur
        records: List[DetectionRecord] = [
            DetectionRecord.from_dict(r, frame_no=frame_no) for r in raw_records
        ]

        persons: List[DetectionRecord] = []
        equipments: List[DetectionRecord] = []
        vehicles: List[DetectionRecord] = []
        hazards: List[DetectionRecord] = []
        others: List[DetectionRecord] = []

        for r in records:
            cls = r.class_name
            obj_id = f"{cls}_{r.track_id}"
            store.update_object(obj_id, r, frame_no=frame_no)

            if cls == CanonicalClass.PERSON.value:
                persons.append(r)
            elif cls in [CanonicalClass.HELMET.value, CanonicalClass.VEST.value]:
                equipments.append(r)
            elif cls == CanonicalClass.VEHICLE.value:
                vehicles.append(r)
            elif cls in [CanonicalClass.FIRE.value, CanonicalClass.SMOKE.value]:
                hazards.append(r)
            else:
                others.append(r)

        frame_warnings: List[str] = []

        # 1. KKD Kontrolleri (Greedy Association: her kare için used_equipment_ids kümesi)
        used_equipment_ids: Set[Any] = set()
        for p in persons:
            kkd_warns = engine.check_kkd(p, equipments, used_equipment_ids=used_equipment_ids)
            frame_warnings.extend(kkd_warns)
            if kkd_warns:
                summary_stats.kkd_violations += len(kkd_warns)

        # 2. Forklift & İş Makinesi Kontrolleri
        for v in vehicles:
            if v.class_name == CanonicalClass.VEHICLE.value:
                prev_v_info = store.get_valid_previous_record(f"forklift_{v.track_id}", frame_no)
                prev_v = prev_v_info[0] if prev_v_info else None
                prev_f_idx = prev_v_info[1] if prev_v_info else None

                forklift_warns = engine.check_forklift_status(
                    v, prev_v, curr_frame=frame_no, prev_frame=prev_f_idx
                )
                frame_warnings.extend(forklift_warns)
                if any("Yan Yatmış" in w for w in forklift_warns):
                    summary_stats.tipping_alerts += 1

        # 3. Tehlikeli Yakınlık Kontrolü
        prox_warns = engine.check_proximity(persons, vehicles)
        frame_warnings.extend(prox_warns)
        summary_stats.proximity_alerts += len(prox_warns)

        # 4. Yangın ve Duman Kontrolü
        hazard_warns = engine.check_fire_and_smoke(hazards)
        frame_warnings.extend(hazard_warns)
        summary_stats.fire_smoke_alerts += len(hazard_warns)

        # Streak güncelleme: süren uyarılar tekrar yazılmaz, bitenler kapanır
        current_keys = set()
        new_warnings = []
        for w in frame_warnings:
            sem_key = _extract_semantic_key(w)
            current_keys.add(sem_key)
            streak = active_streaks.get(sem_key)
            if streak is None:
                active_streaks[sem_key] = {
                    "start": frame_no,
                    "last": frame_no,
                    "frames": 1,
                    "image": image_name,
                    "text": w,
                }
                new_warnings.append(w)
            elif frame_no - streak["last"] <= 1:
                streak["last"] = frame_no
                streak["frames"] += 1
                streak["text"] = w
            else:
                closed_streaks.append(active_streaks.pop(sem_key))
                active_streaks[sem_key] = {
                    "start": frame_no,
                    "last": frame_no,
                    "frames": 1,
                    "image": image_name,
                    "text": w,
                }
                new_warnings.append(w)

        for sem_key in list(active_streaks):
            if sem_key not in current_keys and frame_no - active_streaks[sem_key]["last"] > 1:
                closed_streaks.append(active_streaks.pop(sem_key))

        # Kare Raporu Oluşturma
        summary_stats.total_warnings += len(new_warnings)
        time_sec = frame_no / float(fps)
        m = int(time_sec // 60)
        s_rem = time_sec % 60
        time_code = f"{m:02d}:{s_rem:04.1f}"
        report_lines.append(f"📸 Görüntü: {image_name} [Zaman: {time_code}]")

        counts_summary = (
            f"   Tespit Edilen Varlıklar ({len(records)} Adet): "
            f"{len(persons)} Personel, {len(vehicles)} Araç/Makine, {len(equipments)} KKD Ekipmanı, {len(hazards)} Tehlike"
        )
        if others:
            counts_summary += f", {len(others)} Diğer"
        report_lines.append(counts_summary)

        report_lines.append("   🔍 Görüntü İşleme ve Varlık Betimlemeleri:")
        frame_detections_desc = []
        for r in records:
            cls = r.class_name
            obj_id = f"{cls}_{r.track_id}"
            prev_r_info = store.get_valid_previous_record(obj_id, frame_no)
            prev_r = prev_r_info[0] if prev_r_info else None
            prev_f_idx = prev_r_info[1] if prev_r_info else None

            desc = engine.describe_detection(
                r,
                equipments=equipments,
                prev_record=prev_r,
                curr_frame=frame_no,
                prev_frame=prev_f_idx,
            )
            report_lines.append(f"     - {desc}")
            frame_detections_desc.append(desc)

        # Mekansal Komşuluk Analizi (Nearby Neighbors)
        neighbor_res = engine.analyze_neighbors(persons, vehicles, equipments)
        report_lines.append("   🌐 Mekansal Komşuluk Analizi (Nearby Neighbors):")
        all_neighbors = neighbor_res["person_person"] + neighbor_res["vehicle_vehicle"] + neighbor_res["unattached_equipment"]
        summary_stats.total_neighbors += len(all_neighbors)
        if all_neighbors:
            for n in all_neighbors:
                report_lines.append(f"     - {n}")
        else:
            report_lines.append("     - (Mekansal yakın komşuluk saptanmadı)")

        # Danger Zone (Tehlike Bölgesi) Analizi
        danger_zone_alerts = engine.analyze_danger_zones(persons, vehicles, hazards, store=store, curr_frame=frame_no)
        report_lines.append("   🚨 Danger Zone (Tehlike Bölgesi) Analizi:")
        if danger_zone_alerts:
            for dz in danger_zone_alerts:
                if "KRİTİK DANGER ZONE İHLALİ" in dz:
                    summary_stats.danger_zone_red_breaches += 1
                elif "DANGER ZONE UYARISI" in dz:
                    summary_stats.danger_zone_yellow_alerts += 1
                report_lines.append(f"     - {dz}")
        else:
            report_lines.append("     - ✅ Danger Zone İhlali Yok (Güvenli Çalışma Mesafesi)")

        if new_warnings:
            report_lines.append("   ⚠️ İSG İhlalleri ve Uyarılar:")
            for w in new_warnings:
                w_lines = w.split("\n")
                report_lines.append(f"     - {w_lines[0]}")
                for sub_line in w_lines[1:]:
                    report_lines.append(f"       {sub_line}")
        elif frame_warnings:
            report_lines.append("   ℹ️ İSG Durumu: Önceki karelerde başlatılan ihlal uyarısı devam ediyor.")
        else:
            report_lines.append("   ✅ İSG Durumu: Herhangi bir ihlal algılanmadı (Güvenli).")

        report_lines.append("")

        # JSON çıktısı için kare kaydı
        if output_json_path:
            json_frames.append({
                "frame_no": frame_no,
                "image_name": image_name,
                "timestamp": time_code,
                "counts": {
                    "total": len(records),
                    "person": len(persons),
                    "vehicle": len(vehicles),
                    "equipment": len(equipments),
                    "hazard": len(hazards),
                    "other": len(others),
                },
                "detections": [r.to_dict() for r in records],
                "neighbors": all_neighbors,
                "danger_zone_alerts": danger_zone_alerts,
                "warnings": new_warnings,
                "active_warnings": frame_warnings,
            })

    # Kalan aktif streak'leri kapat
    closed_streaks.extend(active_streaks.values())

    # 1. Plain Text Raporunu Üret ve Kaydet
    full_report = generate_text_report(report_lines, summary_stats, closed_streaks, fps=fps)
    output_path = Path(output_txt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"ISG Raporu basariyla olusturuldu: {output_txt_path}")

    # 2. JSON Raporunu Üret ve Kaydet (İstenmişse)
    if output_json_path:
        json_data = generate_json_report(
            summary_stats, closed_streaks, json_frames, fps=fps, source_csv=str(csv_path)
        )
        json_p = Path(output_json_path)
        json_p.parent.mkdir(parents=True, exist_ok=True)
        with open(json_p, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"ISG JSON Raporu basariyla olusturuldu: {output_json_path}")

    # 3. Markdown Dashboard Raporunu Üret ve Kaydet (İstenmişse)
    if output_md_path:
        md_content = generate_markdown_report(summary_stats, closed_streaks, fps=fps)
        md_p = Path(output_md_path)
        md_p.parent.mkdir(parents=True, exist_ok=True)
        with open(md_p, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"ISG Markdown Raporu basariyla olusturuldu: {output_md_path}")

    return full_report, summary_stats.to_dict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ByteTrack CSV → Çok Formatlı Profesyonel İSG Kural Raporu")
    parser.add_argument("csv", help="ByteTrack tahmin CSV dosyası")
    parser.add_argument("output", help="Çıktı raporu (.txt) yolu")
    parser.add_argument("--fps", type=float, default=30.0, help="Video fps (hız kestirimi için)")
    parser.add_argument("--json", default=None, help="Opsiyonel JSON formatında çıktı raporu yolu")
    parser.add_argument("--markdown", default=None, help="Opsiyonel Markdown formatında dashboard raporu yolu")
    args = parser.parse_args()

    run_isg_analysis(
        args.csv,
        args.output,
        fps=args.fps,
        output_json_path=args.json,
        output_md_path=args.markdown,
    )
