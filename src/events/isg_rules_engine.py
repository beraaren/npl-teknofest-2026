"""
Bu modül, Bytetrack ile tespit edilen nesnelerin kare kare analizini yapar.
.csv formatından veri okuyup, kare kare analiz ederek İSG kurallarına göre rapor oluşturur.
"""
import pandas as pd
import numpy as np
import math
from collections import defaultdict
from typing import List, Dict, Any

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
    
    # Ortalama Fiziksel Boyut Referansları (Metre)
    HUMAN_HEIGHT_M = 1.70
    FORKLIFT_HEIGHT_M = 2.00
    
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

    def estimate_real_distance(self, box1: dict, box2: dict, ref_height_m: float = HUMAN_HEIGHT_M) -> float:
        """
        Nesnelerin piksel cinsinden yüksekliğini gerçek boyuta oranlayarak metre cinsinden mesafesini hesaplar.
        """
        avg_pixel_height = (box1['height'] + box2['height']) / 2.0
        if avg_pixel_height == 0:
            return 999.0
            
        meters_per_pixel = ref_height_m / avg_pixel_height
        pixel_dist = math.hypot(box1['center_x'] - box2['center_x'], box1['center_y'] - box2['center_y'])
        
        return pixel_dist * meters_per_pixel

    def extract_frame_number(self, image_name: str) -> int:
        """
        Görüntü dosya adından kare numarasını (frame index) çıkarmaya çalışır.
        """
        import re
        match = re.search(r'(?:frame_|Frame_No)(\d+)', image_name)
        if match:
            return int(match.group(1))
        return None

    def estimate_speed_kmh(self, curr_box: dict, prev_box: dict, curr_img: str, prev_img: str, ref_height_m: float = HUMAN_HEIGHT_M) -> float:
        """
        İki kare arasındaki piksel kaymasını kare farkı ve zaman değişimine (dt) bölerek km/saat cinsinden hesaplar.
        """
        if prev_box is None or curr_box['height'] == 0:
            return 0.0
            
        f_curr = self.extract_frame_number(curr_img)
        f_prev = self.extract_frame_number(prev_img)
        
        if f_curr is not None and f_prev is not None and f_curr > f_prev:
            frame_diff = f_curr - f_prev
            time_delta = frame_diff / float(self.fps)
        else:
            time_delta = 1.0 / float(self.fps)
            
        meters_per_pixel = ref_height_m / float(curr_box['height'])
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
                    # Baret baş bölgesinde mi? (insan yüksekliğinin üst %40'ı)
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
    def check_forklift_status(self, forklift: dict, prev_forklift: dict = None, curr_img: str = "", prev_img: str = "") -> List[str]:
        warnings = []
        forklift_id = f"Forklift-{forklift['track_id']}"
        
        aspect_ratio = forklift['width'] / float(forklift['height']) if forklift['height'] > 0 else 1.0
        
        # Normal forklift dik durduğunda Genişlik / Yükseklik oranı düşüktür.
        # Yan yattığında genişlik belirgin şekilde yükseklikten fazla olur.
        if aspect_ratio > 1.6:
            warnings.append(f"🚨 KRİTİK ALARM: {forklift_id} Yan Yatmış / Devrilmiş Olabilir! (En/Boy Oranı: {aspect_ratio:.2f})")
            
        # Hız kontrolü
        if prev_forklift is not None:
            speed = self.estimate_speed_kmh(forklift, prev_forklift, curr_img, prev_img, ref_height_m=self.FORKLIFT_HEIGHT_M)
            if speed > 15.0:
                warnings.append(f"⚠️ {forklift_id}: Depo içi aşırı hız yapıyor! (Tahmini Hız: {speed} km/s)")

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
                dist_m = self.estimate_real_distance(p, v, ref_height_m=self.HUMAN_HEIGHT_M)
                
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


def run_isg_analysis(csv_path: str, output_txt_path: str):
    """
    bytetrack_predictions_300.csv dosyasını okuyarak her kare için Türkçe İSG raporları üretir.
    """
    df = pd.read_csv(csv_path)
    engine = ISGRulesEngine(fps=30.0)
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
        "fire_smoke_alerts": 0
    }
    
    report_lines.append("==========================================================")
    report_lines.append("     GÖRÜNTÜ İŞLEME İSG (İŞ SAĞLIĞI VE GÜVENLİĞİ) RAPORU   ")
    report_lines.append("==========================================================\n")
    
    for image_name, frame_df in grouped:
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
                prev_img = prev_v['image_name'] if prev_v else ""
                forklift_warns = engine.check_forklift_status(v, prev_v, curr_img=image_name, prev_img=prev_img)
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

        # Kare Raporu Oluşturma
        if frame_warnings:
            summary_stats["total_warnings"] += len(frame_warnings)
            report_lines.append(f"📸 Görüntü: {image_name}")
            report_lines.append(f"   Tespit Edilen Varlıklar: {len(persons)} Personel, {len(vehicles)} Araç/Makine, {len(equipments)} KKD Ekipmanı")
            for w in frame_warnings:
                report_lines.append(f"   - {w}")
            report_lines.append("")

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
    report_lines.append("==========================================================")
    
    full_report = "\n".join(report_lines)
    
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(full_report)
        
    print(f"ISG Raporu basariyla olusturuldu: {output_txt_path}")
    return full_report, summary_stats

if __name__ == "__main__":
    csv_file = r"c:\Users\buket\Desktop\nlp teknofest\bytetrack_predictions_300.csv"
    output_file = r"c:\Users\buket\Desktop\nlp teknofest\isg_raporu.txt"
    run_isg_analysis(csv_file, output_file)
