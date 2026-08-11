"""
Bu modül, Bytetrack ile tespit edilen nesnelerin kare kare analizini yapar.
.csv formatından veri okuyup, kare kare analiz ederek İSG kurallarına göre rapor oluşturur.

Kural eşikleri (çakışma 0.3, baş bölgesi %45, en/boy 1.6, hız 15 km/h, mesafe 3.0 m)
saha denemeleriyle belirlenmiştir — dokunmayın.
"""
import argparse
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd


class ISGTrackingStore:
    """
    Kareler arası nesne takibi (track_id) ve hareket geçmişi tutan hafıza sınıfı.
    """
    def __init__(self):
        self.history = defaultdict(list)  # obj_id -> list of position records
        self.active_objects = {}          # obj_id -> current record

    def update_object(self, obj_id: str, record: dict):
        self.history[obj_id].append(record)
        self.active_objects[obj_id] = record

    def get_previous_record(self, obj_id: str):
        if len(self.history[obj_id]) >= 2:
            return self.history[obj_id][-2]
        return None


class ISGRulesEngine:
    """
    İş Sağlığı ve Güvenliği (İSG) Kurallar Motoru.
    Geliştirmeye açık if-else yapılarıyla yeni İSG senaryolarının kolayca eklenebileceği modüler mimari.
    """

    # Sınıf bazlı gerçek boyut referansları (Metre) — mesafe/hız kestiriminde ölçek için
    CLASS_HEIGHTS_M = {
        "person": 1.70,
        "forklift": 2.00,
        "machinery": 2.50,
    }
    DEFAULT_HEIGHT_M = 1.70

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    @staticmethod
    def is_inside_or_overlapping(boxA: dict, boxB: dict, threshold: float = 0.3) -> bool:
        """
        boxB'nin (ör. baret/yelek) boxA (ör. insan) ile çakışıp çakışmadığını kontrol eder.
        """
        xA = max(boxA['x1'], boxB['x1'])
        yA = max(boxA['y1'], boxB['y1'])
        xB = min(boxA['x2'], boxB['x2'])
        yB = min(boxA['y2'], boxB['y2'])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxBArea = boxB['area']

        if boxBArea == 0:
            return False

        overlap = interArea / float(boxBArea)
        return overlap >= threshold

    def _meters_per_pixel(self, box1: dict, box2: dict) -> float:
        """
        Ölçeği sınıfı bilinen nesnenin kendi gerçek boyutundan kurar.
        Eskiden iki kutunun ortalama piksel yüksekliği tek referansla (insan)
        ölçekleniyordu; forklift-insan çiftinde forklift'in uzunluğu ölçeği
        bozuyordu. Şimdi insan varsa insan, yoksa sınıfı bilinen diğer nesne
        referans alınır.
        """
        for box in (box1, box2):
            if box.get('class_name') == 'person' and box['height'] > 0:
                return self.CLASS_HEIGHTS_M['person'] / float(box['height'])
        for box in (box1, box2):
            h_m = self.CLASS_HEIGHTS_M.get(box.get('class_name'))
            if h_m and box['height'] > 0:
                return h_m / float(box['height'])
        avg_pixel_height = (box1['height'] + box2['height']) / 2.0
        if avg_pixel_height == 0:
            return 0.0
        return self.DEFAULT_HEIGHT_M / avg_pixel_height

    def estimate_real_distance(self, box1: dict, box2: dict) -> float:
        """
        İki nesne arasındaki mesafeyi metre cinsinden kestirir.
        """
        meters_per_pixel = self._meters_per_pixel(box1, box2)
        if meters_per_pixel == 0:
            return 999.0
        pixel_dist = math.hypot(box1['center_x'] - box2['center_x'], box1['center_y'] - box2['center_y'])
        return pixel_dist * meters_per_pixel

    @staticmethod
    def extract_frame_number(image_name: str) -> Optional[int]:
        """
        Görüntü dosya adından kare numarasını (frame index) çıkarmaya çalışır.
        """
        match = re.search(r'(?:frame_|Frame_No)(\d+)', image_name)
        if match:
            return int(match.group(1))
        return None

    def estimate_speed_kmh(
        self,
        curr_box: dict,
        prev_box: dict,
        curr_frame: Optional[int] = None,
        prev_frame: Optional[int] = None,
    ) -> float:
        """
        İki kare arasındaki piksel kaymasını kare farkı ve zaman değişimine (dt)
        bölerek km/saat cinsinden hesaplar. Kare numaraları doğrudan verilir;
        dosya adı ayrıştırmasına gerek kalmaz.
        """
        if prev_box is None or curr_box['height'] == 0:
            return 0.0

        if curr_frame is not None and prev_frame is not None and curr_frame > prev_frame:
            time_delta = (curr_frame - prev_frame) / float(self.fps)
        else:
            time_delta = 1.0 / float(self.fps)

        meters_per_pixel = self._meters_per_pixel(curr_box, prev_box)
        if meters_per_pixel == 0:
            return 0.0
        pixel_disp = math.hypot(curr_box['center_x'] - prev_box['center_x'], curr_box['center_y'] - prev_box['center_y'])
        distance_m = pixel_disp * meters_per_pixel

        speed_m_per_sec = distance_m / time_delta
        speed_kmh = speed_m_per_sec * 3.6
        return round(speed_kmh, 1)

    # -------------------------------------------------------------
    # İSG KURAL 1: KKD (Baret & Yelek) Kontrolü
    # -------------------------------------------------------------
    def check_kkd(self, person: dict, equipment_boxes: List[dict]) -> List[str]:
        warnings = []
        has_helmet = False
        has_vest = False

        for eq in equipment_boxes:
            if self.is_inside_or_overlapping(person, eq):
                if eq['class_name'] == 'helmet':
                    # Baret baş bölgesinde mi? (insan yüksekliğinin üst %45'i)
                    if eq['center_y'] <= person['y1'] + (person['height'] * 0.45):
                        has_helmet = True
                elif eq['class_name'] == 'vest':
                    has_vest = True

        person_id = f"Personel-{person['track_id']}"

        if not has_helmet and not has_vest:
            warnings.append(f"⚠️ {person_id}: Baret ve Yelek TAKMIYOR! (Kritik İSG İhlali)")
        elif not has_helmet:
            warnings.append(f"⚠️ {person_id}: Baret TAKMIYOR!")
        elif not has_vest:
            warnings.append(f"⚠️ {person_id}: Koruyucu Yelek TAKMIYOR!")

        return warnings

    # -------------------------------------------------------------
    # İSG KURAL 2: Forklift Yan Yatması / Devrilme Kontrolü
    # -------------------------------------------------------------
    def check_forklift_status(
        self,
        forklift: dict,
        prev_forklift: dict = None,
        curr_frame: Optional[int] = None,
        prev_frame: Optional[int] = None,
    ) -> List[str]:
        warnings = []
        forklift_id = f"Forklift-{forklift['track_id']}"

        aspect_ratio = forklift['width'] / float(forklift['height']) if forklift['height'] > 0 else 1.0

        # Normal forklift dik durduğunda Genişlik / Yükseklik oranı düşüktür.
        # Yan yattığında genişlik belirgin şekilde yükseklikten fazla olur.
        if aspect_ratio > 1.6:
            warnings.append(f"🚨 KRİTİK ALARM: {forklift_id} Yan Yatmış / Devrilmiş Olabilir! (En/Boy Oranı: {aspect_ratio:.2f})")

        # Hız kontrolü
        if prev_forklift is not None:
            speed = self.estimate_speed_kmh(forklift, prev_forklift, curr_frame=curr_frame, prev_frame=prev_frame)
            if speed > 15.0:
                warnings.append(f"⚠️ {forklift_id}: Depo içi aşırı hız yapıyor! (Tahmini Hız: {speed} km/h)")

        return warnings

    # -------------------------------------------------------------
    # İSG KURAL 3: Tehlikeli Yakınlık Kontrolü (İnsan - İş Makinesi)
    # -------------------------------------------------------------
    def check_proximity(self, persons: List[dict], vehicles: List[dict], min_safe_meters: float = 3.0) -> List[str]:
        warnings = []
        for p in persons:
            p_id = f"Personel-{p['track_id']}"
            for v in vehicles:
                v_id = f"{v['class_name'].capitalize()}-{v['track_id']}"
                dist_m = self.estimate_real_distance(p, v)

                if dist_m < min_safe_meters:
                    warnings.append(f"🚨 KRİTİK TEHLİKE: {p_id} ile {v_id} tehlikeli derecede yakın! (Mesafe: ~{dist_m:.1f}m)")
        return warnings

    # -------------------------------------------------------------
    # İSG KURAL 4: Yangın ve Duman Kontrolü
    # -------------------------------------------------------------
    def check_fire_and_smoke(self, hazards: List[dict]) -> List[str]:
        warnings = []
        for h in hazards:
            h_id = f"{h['class_name'].capitalize()}-{h['track_id']}"
            if h['class_name'] == 'fire':
                warnings.append(f"🔥 ACİL DURUM: {h_id} (Ateş/Yangın) algılandı! Derhal müdahale edilmeli!")
            elif h['class_name'] == 'smoke':
                warnings.append(f"💨 ACİL UYARI: {h_id} (Duman) algılandı! Yangın riski kontrol edilmeli.")
        return warnings


def run_isg_analysis(csv_path: str, output_txt_path: str, fps: float = 30.0):
    """
    ByteTrack çıktısı CSV'yi okuyarak her kare için Türkçe İSG raporları üretir.

    Aynı uyarı ardışık karelerde sürüyorsa rapora yalnızca ilk karesinde yazılır
    (streak takibi); kaç kare sürdüğü özet bölümünde raporlanır. Böylece tek
    karelik titremeler de, yüzlerce karelik tekrar gürültüsü de elenir.
    """
    df = pd.read_csv(csv_path)
    engine = ISGRulesEngine(fps=fps)
    store = ISGTrackingStore()

    grouped = df.groupby('image_name', sort=False)

    report_lines = []
    summary_stats = {
        "total_images": len(grouped),
        "total_detections": len(df),
        "total_warnings": 0,
        "kkd_violations": 0,
        "tipping_alerts": 0,
        "proximity_alerts": 0,
        "fire_smoke_alerts": 0,
    }

    # Streak takibi: uyarı metni -> {"start": kare_no, "last": kare_no, "frames": n}
    active_streaks: Dict[str, Dict[str, Any]] = {}
    closed_streaks: List[Dict[str, Any]] = []
    frame_no = 0

    report_lines.append("==========================================================")
    report_lines.append("     GÖRÜNTÜ İŞLEME İSG (İŞ SAĞLIĞI VE GÜVENLİĞİ) RAPORU   ")
    report_lines.append("==========================================================\n")

    for image_name, frame_df in grouped:
        frame_no += 1
        records = frame_df.to_dict('records')

        persons = []
        equipments = []
        vehicles = []
        hazards = []

        for r in records:
            cls = r['class_name']
            obj_id = f"{cls}_{r['track_id']}"
            store.update_object(obj_id, r)

            if cls == 'person':
                persons.append(r)
            elif cls in ['helmet', 'vest']:
                equipments.append(r)
            elif cls in ['forklift', 'machinery']:
                vehicles.append(r)
            elif cls in ['fire', 'smoke']:
                hazards.append(r)

        frame_warnings = []

        # 1. KKD Kontrolleri
        for p in persons:
            kkd_warns = engine.check_kkd(p, equipments)
            frame_warnings.extend(kkd_warns)
            if kkd_warns:
                summary_stats["kkd_violations"] += len(kkd_warns)

        # 2. Forklift & İş Makinesi Kontrolleri
        for v in vehicles:
            if v['class_name'] == 'forklift':
                prev_v = store.get_previous_record(f"forklift_{v['track_id']}")
                forklift_warns = engine.check_forklift_status(
                    v, prev_v, curr_frame=frame_no, prev_frame=frame_no - 1 if prev_v else None
                )
                frame_warnings.extend(forklift_warns)
                if any("Yan Yatmış" in w for w in forklift_warns):
                    summary_stats["tipping_alerts"] += 1

        # 3. Tehlikeli Yakınlık Kontrolü
        prox_warns = engine.check_proximity(persons, vehicles)
        frame_warnings.extend(prox_warns)
        summary_stats["proximity_alerts"] += len(prox_warns)

        # 4. Yangın ve Duman Kontrolü
        hazard_warns = engine.check_fire_and_smoke(hazards)
        frame_warnings.extend(hazard_warns)
        summary_stats["fire_smoke_alerts"] += len(hazard_warns)

        # Streak güncelle: süren uyarılar tekrar yazılmaz, bitenler kapanır
        current = set(frame_warnings)
        new_warnings = []
        for w in frame_warnings:
            streak = active_streaks.get(w)
            if streak is None:
                active_streaks[w] = {"start": frame_no, "last": frame_no, "frames": 1, "image": image_name}
                new_warnings.append(w)
            elif frame_no - streak["last"] <= 1:
                streak["last"] = frame_no
                streak["frames"] += 1
            else:
                # Arada kesinti olmuş: eski streak'i kapat, yenisini başlat
                closed_streaks.append({"text": w, **active_streaks.pop(w)})
                active_streaks[w] = {"start": frame_no, "last": frame_no, "frames": 1, "image": image_name}
                new_warnings.append(w)
        for w in list(active_streaks):
            if w not in current and frame_no - active_streaks[w]["last"] > 1:
                closed_streaks.append({"text": w, **active_streaks.pop(w)})

        # Kare Raporu Oluşturma (sadece yeni başlayan uyarılar)
        if new_warnings:
            summary_stats["total_warnings"] += len(new_warnings)
            report_lines.append(f"📸 Görüntü: {image_name}")
            report_lines.append(f"   Tespit Edilen Varlıklar: {len(persons)} Personel, {len(vehicles)} Araç/Makine, {len(equipments)} KKD Ekipmanı")
            for w in new_warnings:
                report_lines.append(f"   - {w}")
            report_lines.append("")

    closed_streaks.extend({"text": w, **s} for w, s in active_streaks.items())

    # Genel Özet İstatistikleri
    report_lines.append("----------------------------------------------------------")
    report_lines.append("                  İSG RAPOR ÖZETİ                         ")
    report_lines.append("----------------------------------------------------------")
    report_lines.append(f"Analiz Edilen Toplam Görüntü Sayısı : {summary_stats['total_images']}")
    report_lines.append(f"Üretilen Toplam İSG Uyarısı Sayısı  : {summary_stats['total_warnings']}")
    report_lines.append(f"  • KKD (Baret/Yelek) İhlali         : {summary_stats['kkd_violations']}")
    report_lines.append(f"  • Forklift Yan Yatması / Devrilme  : {summary_stats['tipping_alerts']}")
    report_lines.append(f"  • Tehlikeli Yakınlık İhlali        : {summary_stats['proximity_alerts']}")
    report_lines.append(f"  • Yangın / Duman İhbarı            : {summary_stats['fire_smoke_alerts']}")

    if closed_streaks:
        report_lines.append("")
        report_lines.append("Uyarı Süreklilik Detayı (kaç kare sürdüğü):")
        for s in sorted(closed_streaks, key=lambda x: -x["frames"]):
            report_lines.append(f"  • {s['text']}  → {s['frames']} kare (ilk: {s['image']})")

    report_lines.append("==========================================================")

    full_report = "\n".join(report_lines)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"ISG Raporu basariyla olusturuldu: {output_txt_path}")
    return full_report, summary_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ByteTrack CSV → İSG kural raporu")
    parser.add_argument("csv", help="ByteTrack tahmin CSV dosyası")
    parser.add_argument("output", help="Çıktı raporu (.txt) yolu")
    parser.add_argument("--fps", type=float, default=30.0, help="Video fps (hız kestirimi için)")
    args = parser.parse_args()
    run_isg_analysis(args.csv, args.output, fps=args.fps)
