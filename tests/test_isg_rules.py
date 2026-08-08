"""ISGRulesEngine testleri — eşikler denenmiş değerlerdir, değiştirme."""
import pandas as pd
import pytest

from src.events.isg_rules_engine import ISGRulesEngine, run_isg_analysis


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
    assert report.count("   - ⚠️ Personel-1: Baret ve Yelek TAKMIYOR") == 1
    assert "3 kare" in report                     # süreklilik detayı
