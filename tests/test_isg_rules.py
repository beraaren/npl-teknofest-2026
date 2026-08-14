"""ISGRulesEngine testleri — eşikler denenmiş değerlerdir, değiştirme."""
import json
from pathlib import Path
import pandas as pd
import pytest

from src.events.isg_rules_engine import (
    BoundingBox,
    CanonicalClass,
    DetectionRecord,
    ISGAlert,
    ISGReportSummary,
    ISGRulesEngine,
    ISGTrackingStore,
    RiskSeverity,
    generate_json_report,
    generate_markdown_report,
    generate_text_report,
    run_isg_analysis,
)


def person(track_id=1, cx=100, cy=100, h=170):
    return {"class_name": "person", "track_id": track_id, "center_x": cx, "center_y": cy,
            "x1": cx - 40, "y1": cy - h // 2, "x2": cx + 40, "y2": cy + h // 2,
            "width": 80, "height": h, "area": 80 * h}


def forklift(track_id=1, cx=500, cy=100, w=80, h=200):
    return {"class_name": "forklift", "track_id": track_id, "center_x": cx, "center_y": cy,
            "x1": cx - w // 2, "y1": cy - h // 2, "x2": cx + w // 2, "y2": cy + h // 2,
            "width": w, "height": h, "area": w * h}


def test_distance_uses_person_scale_not_average():
    """Ölçek insanın kendi boyundan kurulur; forklift'in uzunluğu ortalamayı bozmaz."""
    engine = ISGRulesEngine(fps=30.0)
    p = person(cx=0)
    f = forklift(cx=1000)  # 1000 px yatay mesafe
    dist = engine.estimate_real_distance(p, f)
    # insan 170 px = 1.70 m → 0.01 m/px → ~10 m (güvenli, uyarı yok)
    assert dist == pytest.approx(10.0, abs=0.5)
    warns = engine.check_proximity([p], [f])
    assert warns == []
    # 250 px → 2.5 m < 3.0 m → kritik uyarı
    warns = engine.check_proximity([p], [forklift(cx=250)])
    assert len(warns) == 1 and "TEHLİKE" in warns[0]


def test_speed_kmh_unit_and_frame_numbers():
    """Hız km/h biriminde; kare numaraları doğrudan verilir (dosya adı gerekmez)."""
    engine = ISGRulesEngine(fps=30.0)
    prev = forklift(cx=500)
    curr = forklift(cx=600)  # 100 px kayma
    # forklift 200 px = 2.0 m → 0.01 m/px → 1 m / (15 kare / 30 fps) = 2 m/s = 7.2 km/h
    speed = engine.estimate_speed_kmh(curr, prev, curr_frame=15, prev_frame=0)
    assert speed == pytest.approx(7.2, abs=0.1)
    warns = engine.check_forklift_status(curr, prev, curr_frame=15, prev_frame=0)
    assert not any("km/h" in w for w in warns)  # 7.2 < 15 eşiği


def test_tip_over_threshold_kept():
    engine = ISGRulesEngine()
    tipped = forklift(w=340, h=200)  # en/boy = 1.7 > 1.6
    assert any("Yan Yatmış" in w for w in engine.check_forklift_status(tipped))
    normal = forklift(w=300, h=200)  # 1.5 < 1.6
    assert engine.check_forklift_status(normal) == []


def test_kkd_head_region_rule():
    engine = ISGRulesEngine()
    p = person(cy=200, h=200)  # y1=100, baş bölgesi sınırı y1+90=190
    helmet_ok = {"class_name": "helmet", "track_id": 9, "center_x": 100, "center_y": 120,
                 "x1": 90, "y1": 110, "x2": 110, "y2": 130, "width": 20, "height": 20, "area": 400}
    assert engine.check_kkd(p, [helmet_ok]) == [] or all("Baret TAKMIYOR" not in w for w in engine.check_kkd(p, [helmet_ok]))
    # Baret yok → uyarı
    warns = engine.check_kkd(p, [])
    assert any("Baret" in w for w in warns)


def test_streak_dedup_in_report(tmp_path):
    """Aynı uyarı 3 kare sürerse rapora 1 kez yazılır, özet 3 kare süreklilik gösterir."""
    rows = []
    for i in range(1, 4):
        rows.append({"image_name": f"frame_{i:04d}.jpg", "class_name": "person", "track_id": 1,
                     "center_x": 100, "center_y": 200, "x1": 60, "y1": 100, "x2": 140, "y2": 300,
                     "width": 80, "height": 200, "area": 16000})
    csv_path = tmp_path / "tracks.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    out_path = tmp_path / "rapor.txt"

    report, stats = run_isg_analysis(str(csv_path), str(out_path), fps=30.0)
    assert stats["total_warnings"] == 1          # 3 karede aynı ihlal → 1 uyarı
    # Kare bölümünde yalnızca 1 kez yazılır (süreklilik özeti ayrı satır)
    assert report.count("   - [KRİTİK İHLAL | RİSK SEVİYESİ 4] ⚠️ Personel-1: Baret ve Yelek TAKMIYOR") == 1
    assert "3 kare" in report                     # süreklilik detayı


def test_analyze_neighbors():
    engine = ISGRulesEngine()
    p1 = person(track_id=1, cx=100, cy=100, h=170)
    p2 = person(track_id=2, cx=190, cy=100, h=170)  # 90px = 0.9m < 1.5m
    res = engine.analyze_neighbors([p1, p2], [], [])
    assert len(res["person_person"]) == 1
    assert "Personel-1 ile Personel-2" in res["person_person"][0]


def test_analyze_danger_zones():
    engine = ISGRulesEngine()
    p = person(track_id=1, cx=100, cy=100, h=170)
    f_red = forklift(track_id=1, cx=250, cy=100)  # 150px = 1.5m < 2.0m -> Kırmızı Bölge
    alerts = engine.analyze_danger_zones([p], [f_red], [])
    assert len(alerts) == 1
    assert "KRİTİK DANGER ZONE İHLALİ" in alerts[0]


def test_turkish_class_normalization():
    """Türkçe etiketler ('insan', 'baret', 'yelek', 'ates', 'duman') sorunsuz işlenmelidir."""
    engine = ISGRulesEngine()
    p_tr = {"class_name": "insan", "track_id": 1, "center_x": 100, "center_y": 100,
            "x1": 60, "y1": 15, "x2": 140, "y2": 185, "width": 80, "height": 170, "area": 13600}
    baret_tr = {"class_name": "baret", "track_id": 2, "center_x": 100, "center_y": 30,
                "x1": 85, "y1": 20, "x2": 115, "y2": 45, "width": 30, "height": 25, "area": 750}
    yelek_tr = {"class_name": "yelek", "track_id": 3, "center_x": 100, "center_y": 80,
                "x1": 75, "y1": 50, "x2": 125, "y2": 120, "width": 50, "height": 70, "area": 3500}

    # Baret ve Yelek var -> KKD ihlali olmamalı
    warns = engine.check_kkd(p_tr, [baret_tr, yelek_tr])
    assert len(warns) == 0

    # Ateş ve Duman kontrolü
    hazards_tr = [
        {"class_name": "yangin", "track_id": 1, "center_x": 300, "center_y": 300, "x1": 280, "y1": 280, "x2": 320, "y2": 320, "width": 40, "height": 40, "area": 1600},
        {"class_name": "duman", "track_id": 2, "center_x": 350, "center_y": 300, "x1": 330, "y1": 280, "x2": 370, "y2": 320, "width": 40, "height": 40, "area": 1600},
    ]
    hazard_warns = engine.check_fire_and_smoke(hazards_tr)
    assert len(hazard_warns) == 2
    assert any("Ateş/Yangın" in w for w in hazard_warns)
    assert any("Duman" in w for w in hazard_warns)


def test_multiformat_report_generation_json_and_markdown(tmp_path):
    """JSON ve Markdown rapor çıktılarının eksiksiz üretildiğini doğrular."""
    rows = [
        {"image_name": "frame_0001.jpg", "class_name": "person", "track_id": 1, "x1": 100, "y1": 100, "x2": 180, "y2": 270, "confidence": 0.95},
        {"image_name": "frame_0001.jpg", "class_name": "forklift", "track_id": 2, "x1": 200, "y1": 100, "x2": 280, "y2": 300, "confidence": 0.90},
    ]
    csv_path = tmp_path / "test.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    txt_path = tmp_path / "out.txt"
    json_path = tmp_path / "out.json"
    md_path = tmp_path / "out.md"

    report, stats = run_isg_analysis(
        str(csv_path),
        str(txt_path),
        fps=30.0,
        output_json_path=str(json_path),
        output_md_path=str(md_path),
    )

    assert txt_path.exists()
    assert json_path.exists()
    assert md_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "metadata" in data
        assert "summary_stats" in data
        assert data["summary_stats"]["total_images"] == 1
        assert len(data["frames"]) == 1

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
        assert "# 🛡️ BERA İSG (İş Sağlığı ve Güvenliği) Analiz Raporu" in md_content
        assert "ISO 45001" in md_content


def test_tracking_store_reacquisition_gap():
    """Uzun süre görünmeyen nesne geri geldiğinde sahte hız sıçraması olmamalıdır."""
    store = ISGTrackingStore()
    rec1 = {"class_name": "forklift", "track_id": 1, "center_x": 100, "center_y": 100, "width": 80, "height": 200, "x1": 60, "y1": 0, "x2": 140, "y2": 200}
    rec2 = {"class_name": "forklift", "track_id": 1, "center_x": 400, "center_y": 100, "width": 80, "height": 200, "x1": 360, "y1": 0, "x2": 440, "y2": 200}

    store.update_object("forklift_1", rec1, frame_no=1)
    store.update_object("forklift_1", rec2, frame_no=100)  # 99 kare sonra (yaklaşık 3.3 saniye)

    # 10 karelik maksimum boşluk eşiğinde None dönmeli (sahte hız sıçramasını engeller)
    valid = store.get_valid_previous_record("forklift_1", current_frame=100, max_frame_gap=10)
    assert valid is None

    # Aradaki boşluk 2 kare iken geçerli dönmeli
    store2 = ISGTrackingStore()
    store2.update_object("forklift_1", rec1, frame_no=10)
    store2.update_object("forklift_1", rec2, frame_no=12)
    valid2 = store2.get_valid_previous_record("forklift_1", current_frame=12, max_frame_gap=10)
    assert valid2 is not None
    assert valid2[1] == 10


def test_bounding_box_dataclass():
    """BoundingBox güvenli koordinat, alan, merkez ve en/boy hesaplayıcılarını test eder."""
    bbox = BoundingBox(x1=100.0, y1=50.0, x2=20.0, y2=10.0)  # ters koordinat
    assert bbox.x1 == 20.0 and bbox.x2 == 100.0
    assert bbox.y1 == 10.0 and bbox.y2 == 50.0
    assert bbox.width == 80.0
    assert bbox.height == 40.0
    assert bbox.area == 3200.0
    assert bbox.center_x == 60.0
    assert bbox.center_y == 30.0
    assert bbox.aspect_ratio == 2.0
    assert bbox.safe_xyxy == (20.0, 10.0, 100.0, 50.0)

    # from_dict ve to_dict
    bbox2 = BoundingBox.from_dict({"x1": 10, "y1": 20, "width": 30, "height": 40})
    assert bbox2.x2 == 40.0 and bbox2.y2 == 60.0
    assert bbox2.area == 1200.0


def test_detection_record_with_bbox():
    """DetectionRecord veri modeli ve geriye dönük uyumlu erişimleri test eder."""
    rec = DetectionRecord.from_dict({
        "image_name": "frame_001.jpg",
        "class_name": "isci",
        "track_id": 5,
        "x1": 50, "y1": 50, "x2": 150, "y2": 250,
        "confidence": 0.88,
    }, frame_no=1)

    assert rec.class_name == CanonicalClass.PERSON.value
    assert rec.width == 100.0
    assert rec.height == 200.0
    assert rec.center_x == 100.0
    assert rec.center_y == 150.0
    # Sözlük erişimi
    assert rec["x1"] == 50.0
    assert rec.get("confidence") == 0.88
    d = rec.to_dict()
    assert d["track_id"] == 5 and d["class_name"] == "person"


def test_isg_alert_and_summary_dataclasses():
    """ISGAlert ve ISGReportSummary veri modellerini test eder."""
    alert = ISGAlert(
        severity=RiskSeverity.CRITICAL,
        category="YANGIN",
        affected_track_ids=[1],
        detail="[ACİL DURUM | RİSK SEVİYESİ 5] 🔥 Yangın algılandı!",
        action="Saha tahliye edilmelidir.",
        time_code="00:05.0",
        frame_idx=150,
    )
    assert alert.severity == RiskSeverity.CRITICAL
    assert "Saha tahliye" in alert.formatted_message()
    alert_dict = alert.to_dict()
    assert alert_dict["severity"] == 5
    assert alert_dict["category"] == "YANGIN"

    summary = ISGReportSummary(total_images=10, total_detections=45)
    summary["kkd_violations"] = 3
    assert summary["kkd_violations"] == 3
    assert summary.total_images == 10
    assert summary.to_dict()["total_detections"] == 45


def test_greedy_ppe_association():
    """Tek bir baretin iki farklı personele aynı anda atanmasını (mükerrer atama) engeller."""
    engine = ISGRulesEngine()
    # 2 personel yan yana
    p1 = person(track_id=1, cx=100, cy=100, h=170)
    p2 = person(track_id=2, cx=105, cy=100, h=170)  # neredeyse aynı konum

    # Sadece 1 adet baret var
    single_helmet = {"class_name": "helmet", "track_id": 99, "center_x": 100, "center_y": 30,
                     "x1": 85, "y1": 20, "x2": 115, "y2": 45, "width": 30, "height": 25, "area": 750}

    used_eq_ids = set()
    warns1 = engine.check_kkd(p1, [single_helmet], used_equipment_ids=used_eq_ids)
    warns2 = engine.check_kkd(p2, [single_helmet], used_equipment_ids=used_eq_ids)

    # Personel 1 bareti aldı (sadece yelek uyarısı alır)
    assert all("Baret TAKMIYOR" not in w for w in warns1)
    # Personel 2 bareti alamaz (kullanıldı) -> Baret ve Yelek uyarısı alır
    assert any("Baret" in w for w in warns2)


def test_standalone_report_generators():
    """generate_text_report, generate_json_report ve generate_markdown_report fonksiyonlarını test eder."""
    summary = ISGReportSummary(
        total_images=5,
        total_detections=20,
        total_warnings=2,
        kkd_violations=2,
    )
    streaks = [{
        "text": "[KRİTİK İHLAL | RİSK SEVİYESİ 4] ⚠️ Personel-1: Baret TAKMIYOR!\n     └─ 💡 Önerilen Aksiyon: Baret verilmeli",
        "frames": 10,
        "image": "frame_0001.jpg",
    }]

    txt = generate_text_report(["📸 Frame 1"], summary, streaks, fps=30.0)
    assert "İSG RAPOR ÖZETİ" in txt
    assert "Analiz Edilen Toplam Görüntü Sayısı : 5" in txt

    json_data = generate_json_report(summary, streaks, [], fps=30.0, source_csv="test.csv")
    assert json_data["summary_stats"]["kkd_violations"] == 2
    assert len(json_data["incident_streaks"]) == 1
    assert json_data["incident_streaks"][0]["duration_frames"] == 10

    md = generate_markdown_report(summary, streaks, fps=30.0)
    assert "# 🛡️ BERA İSG (İş Sağlığı ve Güvenliği) Analiz Raporu" in md
    assert "| **KKD (Baret/Yelek) İhlali** | `2` |" in md

