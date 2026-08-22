"""Olay tespit motoru testleri."""
from src.config import EventsConfig
from src.events.event_engine import EventEngine
from src.perception.detector import Detection
from src.perception.tracker import TrackedObject


def make_track(class_name: str, x1: float, y1: float, x2: float, y2: float, tid: int = 1) -> TrackedObject:
    det = Detection(class_name=class_name, confidence=0.8, bbox=(x1, y1, x2, y2))
    return TrackedObject(track_id=tid, class_name=class_name, initial_detection=det)


def test_gathering_detection():
    cfg = EventsConfig(enabled_rules=["gathering"], thresholds={"gathering": {"min_persons": 3, "max_inter_center_distance": 120}})
    engine = EventEngine(cfg, fps=25.0)

    obs = {
        "frame_idx": 10,
        "timestamp": 0.4,
        "detections": [],
        "tracks": [
            {"track_id": 1, "class": "insan", "history_length": 1, "last_center": [50, 50], "speed": [0, 0]},
            {"track_id": 2, "class": "insan", "history_length": 1, "last_center": [55, 55], "speed": [0, 0]},
            {"track_id": 3, "class": "insan", "history_length": 1, "last_center": [60, 60], "speed": [0, 0]},
        ],
        "scene_graph": {
            "frame_idx": 10,
            "timestamp": 0.4,
            "nodes": [
                {"id": "insan_0", "class": "insan", "track_id": 1, "center": [50, 50], "confidence": 0.8},
                {"id": "insan_1", "class": "insan", "track_id": 2, "center": [55, 55], "confidence": 0.8},
                {"id": "insan_2", "class": "insan", "track_id": 3, "center": [60, 60], "confidence": 0.8},
            ],
            "edges": [],
        },
    }

    signals = engine.process_observation(obs)
    assert len(signals) >= 1
    assert signals[0].event_type == "gathering"


def test_proximity_detection():
    cfg = EventsConfig(enabled_rules=["proximity"], thresholds={"proximity": {"dangerous_pairs": [["forklift", "insan"]], "distance_threshold_pixels": 100}})
    engine = EventEngine(cfg, fps=25.0)

    obs = {
        "frame_idx": 5,
        "timestamp": 0.2,
        "detections": [],
        "tracks": [
            {"track_id": 1, "class": "forklift", "history_length": 1, "last_center": [100, 100], "speed": [0, 0]},
            {"track_id": 2, "class": "insan", "history_length": 1, "last_center": [110, 110], "speed": [0, 0]},
        ],
        "scene_graph": {
            "frame_idx": 5,
            "timestamp": 0.2,
            "nodes": [
                {"id": "forklift_0", "class": "forklift", "track_id": 1, "center": [100, 100], "confidence": 0.9},
                {"id": "insan_0", "class": "insan", "track_id": 2, "center": [110, 110], "confidence": 0.85},
            ],
            "edges": [
                {"source": "forklift_0", "target": "insan_0", "relation": "near", "weight": 0.9}
            ],
        },
    }

    signals = engine.process_observation(obs)
    assert any(s.event_type == "dangerous_proximity" for s in signals)


def _make_person_obs(frame_idx: int, timestamp: float, y1: float, y2: float, x1: float = 100, x2: float = 140, tid: int = 1):
    return {
        "frame_idx": frame_idx,
        "timestamp": timestamp,
        "detections": [
            {"class": "insan", "track_id": tid, "confidence": 0.9, "bbox": [x1, y1, x2, y2], "frame_idx": frame_idx},
        ],
        "tracks": [
            {"track_id": tid, "class": "insan", "history_length": 1, "last_center": [(x1 + x2) / 2, (y1 + y2) / 2], "speed": [0, 0]},
        ],
        "scene_graph": {"frame_idx": frame_idx, "timestamp": timestamp, "nodes": [], "edges": []},
    }


def test_fall_detection_across_frames():
    """Track geçmişi kareler arasında korunmalı, aksi halde speed hep 0 kalır
    ve person_fall hiçbir zaman tetiklenemez (bkz. EventEngine._observation_to_tracks).

    Gerçek düşüşler genelde birden fazla kareye yayılır (bkz. rules.py._rule_fall
    docstring'i); bu yüzden pencereli+oranlı mantık test edilir: window_seconds=0.2,
    fps=10.0 -> window_frames=2, yani 3. karedeki (index 2) displacement,
    0. karenin merkezine göre hesaplanır.
    """
    cfg = EventsConfig(
        enabled_rules=["fall"],
        thresholds={"fall": {"window_seconds": 0.2, "min_drop_ratio": 0.4}},
    )
    engine = EventEngine(cfg, fps=10.0)

    # Sabit yükseklik (h=100) -> scale_ema=100 sabit kalır. cy: 150 -> 150 -> 250
    # (0. kareden 2. kareye toplam 100px düşüş / scale 100 = oran 1.0 >= 0.4)
    engine.process_observation(_make_person_obs(0, 0.0, 100, 200))
    engine.process_observation(_make_person_obs(1, 0.1, 100, 200))
    signals = engine.process_observation(_make_person_obs(2, 0.2, 200, 300))

    assert any(s.event_type == "person_fall" for s in signals), (
        "person_fall sinyali üretilmedi; track geçmişi korunmuyor veya pencere/oran mantığı bozuk olabilir."
    )


def test_fall_not_triggered_for_small_relative_motion():
    """Ölçeğe göre küçük bir hareket (oran eşiğin altında) sinyal üretmemeli."""
    cfg = EventsConfig(
        enabled_rules=["fall"],
        thresholds={"fall": {"window_seconds": 0.2, "min_drop_ratio": 0.4}},
    )
    engine = EventEngine(cfg, fps=10.0)

    # h=100 sabit, cy: 150 -> 150 -> 170 (toplam 20px / scale 100 = oran 0.2 < 0.4)
    engine.process_observation(_make_person_obs(0, 0.0, 100, 200))
    engine.process_observation(_make_person_obs(1, 0.1, 100, 200))
    signals = engine.process_observation(_make_person_obs(2, 0.2, 120, 220))

    assert not any(s.event_type == "person_fall" for s in signals)


def test_fall_is_scale_invariant():
    """Aynı ORANDA düşüş, farklı bbox ölçeklerinde (kameraya yakın/uzak) aynı
    şekilde tetiklenmeli — sabit piksel eşiği olsaydı küçük ölçekli track
    tetiklenemezdi (bu tam olarak ertelenen 'mesafe/derinlik' sorunudur)."""
    cfg = EventsConfig(
        enabled_rules=["fall"],
        thresholds={"fall": {"window_seconds": 0.2, "min_drop_ratio": 0.4}},
    )

    # Küçük ölçek (uzak track): h=50, toplam düşüş 25px -> oran 0.5
    engine_small = EventEngine(cfg, fps=10.0)
    engine_small.process_observation(_make_person_obs(0, 0.0, 100, 150))
    engine_small.process_observation(_make_person_obs(1, 0.1, 100, 150))
    signals_small = engine_small.process_observation(_make_person_obs(2, 0.2, 125, 175))

    # Büyük ölçek (yakın track): h=300, toplam düşüş 150px (aynı ORAN 0.5)
    engine_large = EventEngine(cfg, fps=10.0)
    engine_large.process_observation(_make_person_obs(0, 0.0, 100, 400))
    engine_large.process_observation(_make_person_obs(1, 0.1, 100, 400))
    signals_large = engine_large.process_observation(_make_person_obs(2, 0.2, 250, 550))

    assert any(s.event_type == "person_fall" for s in signals_small), "Küçük ölçekli track tetiklenmedi"
    assert any(s.event_type == "person_fall" for s in signals_large), "Büyük ölçekli track tetiklenmedi"



def _make_hazard_obs(frame_idx: int, timestamp: float, hazard_class: str, confidence: float = 0.9, tid: int = 7):
    return {
        "frame_idx": frame_idx,
        "timestamp": timestamp,
        "detections": [
            {"class": hazard_class, "track_id": tid, "confidence": confidence,
             "bbox": [200, 200, 260, 280], "frame_idx": frame_idx},
        ],
        "tracks": [
            {"track_id": tid, "class": hazard_class, "history_length": 1, "last_center": [230, 240], "speed": [0, 0]},
        ],
        "scene_graph": {"frame_idx": frame_idx, "timestamp": timestamp, "nodes": [], "edges": []},
    }


def test_fire_smoke_detection_after_min_duration():
    """yangin/duman sınıfı min_duration_frames boyunca görülünce sinyal üretmeli."""
    cfg = EventsConfig(
        enabled_rules=["fire_smoke"],
        thresholds={"fire_smoke": {"min_duration_frames": 3, "min_confidence": 0.35}},
    )
    engine = EventEngine(cfg, fps=10.0)

    # İlk 2 kare eşiğin altında (henüz süreklilik yok)
    assert not engine.process_observation(_make_hazard_obs(0, 0.0, "yangin"))
    assert not engine.process_observation(_make_hazard_obs(1, 0.1, "yangin"))
    # 3. karede history_length=3 -> tetiklenmeli
    signals = engine.process_observation(_make_hazard_obs(2, 0.2, "yangin"))

    assert any(s.event_type == "fire_smoke" for s in signals)
    sig = next(s for s in signals if s.event_type == "fire_smoke")
    assert sig.metadata["hazard_class"] == "yangin"


def test_smoke_class_also_triggers_fire_smoke():
    """duman sınıfı da aynı fire_smoke olay tipini üretmeli (risk_patterns.yaml uyumu)."""
    cfg = EventsConfig(
        enabled_rules=["fire_smoke"],
        thresholds={"fire_smoke": {"min_duration_frames": 2, "min_confidence": 0.35}},
    )
    engine = EventEngine(cfg, fps=10.0)

    engine.process_observation(_make_hazard_obs(0, 0.0, "duman"))
    signals = engine.process_observation(_make_hazard_obs(1, 0.1, "duman"))

    assert any(s.event_type == "fire_smoke" for s in signals)
    assert next(s for s in signals if s.event_type == "fire_smoke").metadata["hazard_class"] == "duman"


def test_fire_smoke_low_confidence_filtered():
    """min_confidence altındaki alev/duman tespitleri sinyal üretmemeli."""
    cfg = EventsConfig(
        enabled_rules=["fire_smoke"],
        thresholds={"fire_smoke": {"min_duration_frames": 2, "min_confidence": 0.5}},
    )
    engine = EventEngine(cfg, fps=10.0)

    engine.process_observation(_make_hazard_obs(0, 0.0, "yangin", confidence=0.2))
    signals = engine.process_observation(_make_hazard_obs(1, 0.1, "yangin", confidence=0.2))

    assert not any(s.event_type == "fire_smoke" for s in signals)


def test_fire_smoke_disabled_when_rule_not_enabled():
    """enabled_rules'ta fire_smoke yoksa hiç değerlendirilmemeli."""
    cfg = EventsConfig(enabled_rules=["fall"], thresholds={"fire_smoke": {"min_duration_frames": 1}})
    engine = EventEngine(cfg, fps=10.0)

    engine.process_observation(_make_hazard_obs(0, 0.0, "yangin"))
    signals = engine.process_observation(_make_hazard_obs(1, 0.1, "yangin"))

    assert not any(s.event_type == "fire_smoke" for s in signals)
