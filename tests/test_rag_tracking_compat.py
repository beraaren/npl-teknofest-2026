"""
RAG katmanı ↔ TrackedObject.to_dict() uyumluluk testleri.

Son commit (ByteTrack / track_id entegrasyonu) TrackedObject.to_dict() formatını
değiştirdi: artık 'history' / 'velocity' yerine 'last_center' ve 'speed' var.
Bu testler bu uyumu doğrular.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# Proje kök dizinini path'e ekle
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.reasoning.rag_layer import RAGLayer, _observations_to_natural_language  # noqa: E402


# ---------------------------------------------------------------------------
# Yardımcı: yeni TrackedObject.to_dict() çıktı formatı
# ---------------------------------------------------------------------------
def _make_obs(
    class_name: str = "arac",
    speed: list[float] | None = None,
    last_center: list[float] | None = None,
    bbox: list[float] | None = None,
    track_id: int = 1,
    aspect_ratio_w_h: tuple[float, float] | None = None,
) -> dict:
    """ObserverAgent.observe_frame() çıktısını simüle eder (yeni format)."""
    if bbox is None:
        if aspect_ratio_w_h:
            w, h = aspect_ratio_w_h
            bbox = [100.0, 100.0, 100.0 + w, 100.0 + h]
        else:
            bbox = [100.0, 100.0, 200.0, 200.0]

    if last_center is None:
        last_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

    if speed is None:
        speed = [0.0, 0.0]

    det = {
        "class": class_name,
        "track_id": track_id,
        "confidence": 0.9,
        "bbox": bbox,
        "center": last_center,
        "frame_idx": 0,
    }
    trk = {
        "track_id": track_id,
        "class": class_name,
        "history_length": 5,
        "last_center": last_center,
        "speed": speed,
    }
    return {
        "frame_idx": 0,
        "timestamp": 0.0,
        "detections": [det],
        "tracks": [trk],
        "scene_graph": {"frame_idx": 0, "timestamp": 0.0, "nodes": [], "edges": []},
    }


# ---------------------------------------------------------------------------
# Test 1: Hareket tespiti — speed alanından
# ---------------------------------------------------------------------------
def test_motion_note_from_speed_field():
    """speed=[10, 5] olan track 'hareketli' olarak yakalanmalı."""
    obs = _make_obs(class_name="insan", speed=[10.0, 5.0])
    result = _observations_to_natural_language([obs])
    assert "hareketli" in result.lower(), f"Beklenen 'hareketli', alınan: {result}"


def test_stationary_note_when_speed_zero():
    """speed=[0, 0] olan track için 'hareketsiz' ifadesi çıkmalı."""
    obs = _make_obs(class_name="insan", speed=[0.0, 0.0])
    result = _observations_to_natural_language([obs])
    assert "hareketsiz" in result.lower(), f"Beklenen 'hareketsiz', alınan: {result}"


# ---------------------------------------------------------------------------
# Test 2: Kinematik kontrol — last_center + speed
# ---------------------------------------------------------------------------
def test_kinematic_high_relative_speed_detected():
    """Yakın forklift & insan çifti + yüksek göreceli hız → uyarı satırı."""
    obs_forklift = _make_obs(
        class_name="arac",
        speed=[50.0, 0.0],
        last_center=[200.0, 200.0],
        bbox=[150.0, 150.0, 250.0, 250.0],
        track_id=1,
    )
    obs_person = _make_obs(
        class_name="insan",
        speed=[0.0, 0.0],
        last_center=[220.0, 200.0],
        bbox=[195.0, 150.0, 245.0, 250.0],
        track_id=2,
    )
    # Tek bir observation içinde her ikisini birleştir
    combined = {
        "frame_idx": 0,
        "timestamp": 0.0,
        "detections": obs_forklift["detections"] + obs_person["detections"],
        "tracks": obs_forklift["tracks"] + obs_person["tracks"],
        "scene_graph": obs_forklift["scene_graph"],
    }
    result = _observations_to_natural_language([combined])
    assert "YÜKSEK GÖRECELİ HIZ" in result or "çarpışma" in result.lower(), (
        f"Yüksek göreceli hız uyarısı beklendi, alınan: {result}"
    )


def test_kinematic_uses_last_center_not_history():
    """Eski 'history' anahtarı olmayan track dict'te hata olmamalı."""
    obs = _make_obs(class_name="insan", speed=[2.0, 1.0])
    # 'history' anahtarı kesinlikle yok
    for trk in obs["tracks"]:
        assert "history" not in trk, "Test geçersiz: 'history' anahtarı mevcut"
    # Çalışmalı, hata fırlatmamalı
    result = _observations_to_natural_language([obs])
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Test 3: Aspect ratio anomali tespiti (bbox üzerinden)
# ---------------------------------------------------------------------------
def test_aspect_ratio_fall_detection():
    """Yatay insan bbox (w/h > 1.2) → düşme uyarısı."""
    obs = _make_obs(class_name="insan", aspect_ratio_w_h=(200.0, 80.0))
    result = _observations_to_natural_language([obs])
    assert "yatay pozisyon" in result.lower() or "düşme" in result.lower(), (
        f"Beklenen yatay/düşme uyarısı, alınan: {result}"
    )


def test_aspect_ratio_vehicle_tip_over():
    """Yatay araç bbox (w/h > 1.5) → devrilme uyarısı."""
    obs = _make_obs(class_name="arac", aspect_ratio_w_h=(250.0, 80.0))
    result = _observations_to_natural_language([obs])
    assert "devrilme" in result.lower(), f"Beklenen devrilme uyarısı, alınan: {result}"


# ---------------------------------------------------------------------------
# Test 4: match_patterns() — EventSignal.to_dict() formatı
# ---------------------------------------------------------------------------
def test_match_patterns_with_dict_signals():
    """match_patterns() hem dict hem nesne sinyalini kabul etmeli."""
    rag = RAGLayer()  # patterns dosyası yoksa boş çalışır, hata vermemeli
    signals_as_dict = [
        {
            "event_type": "forklift_tip_over",
            "timestamp": "00:05",
            "description": "test",
            "confidence": 0.9,
            "involved_track_ids": [1, 2],
            "metadata": {},
        }
    ]
    # Hata fırlatmamalı
    matched = rag.match_patterns(signals_as_dict, observation_report=None)
    assert isinstance(matched, dict)


def test_match_patterns_tracks_included_in_structural():
    """Yapısal eşleşmede tracks'teki class da dikkate alınmalı."""
    rag = RAGLayer()

    obs = _make_obs(class_name="arac")
    # detections'ı boşalt ama track'te class var
    obs["detections"] = []

    # Hata fırlatmamalı
    matched = rag.match_patterns([], observation_report=[obs])
    assert isinstance(matched, dict)


# ---------------------------------------------------------------------------
# Test 5: recommend_tools() — stop_forklift için involved_track_ids
# ---------------------------------------------------------------------------
def test_stop_forklift_uses_involved_track_ids():
    """stop_forklift aracı, EventSignal.to_dict()'teki involved_track_ids'ten ID almalı."""
    rag = RAGLayer()

    matched_patterns = [
        {
            "pattern": "dangerous_proximity",
            "description": "Tehlikeli yakınlık",
            "risk_score": 80,
            "risk_level": "Yüksek",
            "potential_hazards": [],
            "similarity": 0.9,
            "matched_signal": {
                "event_type": "dangerous_proximity",
                "timestamp": "00:10",
                "involved_track_ids": [42, 7],
                "metadata": {},
            },
        }
    ]

    # RAG patterns dosyası yoksa mock_tool_hints boş gelir; doğrudan inject et
    rag.patterns = {
        "patterns": {
            "dangerous_proximity": {
                "description": "Tehlikeli yakınlık",
                "risk_score": 80,
                "risk_level": "Yüksek",
                "potential_hazards": [],
                "mock_tool_hints": ["stop_forklift"],
            }
        }
    }

    tools = rag.recommend_tools(matched_patterns, enabled_tools=None, max_tools=5)
    stop_tool = next((t for t in tools if t["tool_name"] == "stop_forklift"), None)

    assert stop_tool is not None, "stop_forklift aracı üretilmedi"
    fid = stop_tool["params"].get("forklift_id")
    # İlk track_id olan 42 bekleniyor
    assert str(fid) == "42", f"Beklenen forklift_id='42', alınan: {fid!r}"


# ---------------------------------------------------------------------------
# Test 6: Boş observations — çökmeme
# ---------------------------------------------------------------------------
def test_empty_observations_no_crash():
    result = _observations_to_natural_language([])
    assert result == ""

    result2 = _observations_to_natural_language([{}])
    assert isinstance(result2, str)


if __name__ == "__main__":
    tests = [
        test_motion_note_from_speed_field,
        test_stationary_note_when_speed_zero,
        test_kinematic_high_relative_speed_detected,
        test_kinematic_uses_last_center_not_history,
        test_aspect_ratio_fall_detection,
        test_aspect_ratio_vehicle_tip_over,
        test_match_patterns_with_dict_signals,
        test_match_patterns_tracks_included_in_structural,
        test_stop_forklift_uses_involved_track_ids,
        test_empty_observations_no_crash,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except Exception as exc:
            print(f"  ✗ {t.__name__}: {exc}")
            failed += 1
    print(f"\n{'TÜMÜ GEÇTİ' if not failed else f'{failed} TEST BAŞARISIZ'} ({len(tests) - failed}/{len(tests)})")
    sys.exit(failed)
