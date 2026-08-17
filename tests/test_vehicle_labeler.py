"""VehicleLabeler testleri — mock VLM backend ile (gerçek modele gerek yok)."""
import json
from types import SimpleNamespace

import numpy as np

from src.perception.detector import Detection
from src.perception.tracker import TrackedObject
from src.perception.vehicle_labeler import (
    VEHICLE_TYPES,
    apply_vehicle_labels,
    build_labeling_prompt,
    collect_vehicle_crops,
    label_vehicles,
)


class MockBackend:
    """VLMBackend.generate() sözleşmesini taklit eder; hazır yanıt döner."""

    def __init__(self, response: str = "{}", raise_exc: bool = False):
        self.response = response
        self.raise_exc = raise_exc
        self.calls = []

    def generate(self, images, prompt, temperature=0.15, max_tokens=1024):
        self.calls.append({"n_images": len(images), "prompt": prompt})
        if self.raise_exc:
            raise RuntimeError("VLM çöktü (simülasyon)")
        return self.response


def make_track(track_id: int, class_name: str = "arac", confidence: float = 0.9,
               bbox=(10, 10, 60, 60), frame_idx: int = 0) -> TrackedObject:
    det = Detection(class_name=class_name, confidence=confidence, bbox=bbox, frame_idx=frame_idx)
    det.track_id = track_id
    return TrackedObject(track_id=track_id, class_name=class_name, initial_detection=det)


def make_frames(n: int = 2, size: int = 100) -> list:
    return [np.zeros((size, size, 3), dtype=np.uint8) for _ in range(n)]


def make_config(**overrides) -> SimpleNamespace:
    base = dict(enabled=True, max_vehicles=8, min_confidence=0.35,
                padding_ratio=0.15, max_tokens=768)
    base.update(overrides)
    return SimpleNamespace(**base)


# ----------------------------------------------------------------------
# collect_vehicle_crops
# ----------------------------------------------------------------------

class TestCollectVehicleCrops:
    def test_arac_track_temsili_crop_uretir(self):
        tracks = {1: make_track(1)}
        crops = collect_vehicle_crops(tracks, make_frames())
        assert len(crops) == 1
        assert crops[0]["track_id"] == 1
        assert crops[0]["crop"].size > 0

    def test_arac_olmayan_siniflar_elenir(self):
        tracks = {1: make_track(1, class_name="insan"), 2: make_track(2)}
        crops = collect_vehicle_crops(tracks, make_frames())
        assert [c["track_id"] for c in crops] == [2]

    def test_dusuk_confidence_elenir(self):
        tracks = {1: make_track(1, confidence=0.10)}
        crops = collect_vehicle_crops(tracks, make_frames(), min_confidence=0.35)
        assert crops == []

    def test_max_vehicles_siniri(self):
        tracks = {i: make_track(i, confidence=0.9 - i * 0.01) for i in range(10)}
        crops = collect_vehicle_crops(tracks, make_frames(), max_vehicles=3)
        assert len(crops) == 3

    def test_historyden_en_iyi_temsil_secilir(self):
        track = make_track(1, confidence=0.5, frame_idx=0)
        better = Detection(class_name="arac", confidence=0.95, bbox=(0, 0, 50, 50), frame_idx=1)
        better.track_id = 1
        track.update(better)
        crops = collect_vehicle_crops({1: track}, make_frames(n=3))
        assert crops[0]["frame_idx"] == 1
        assert crops[0]["yolo_confidence"] == 0.95

    def test_frame_idx_aralik_disi_atlanir(self):
        tracks = {1: make_track(1, frame_idx=99)}
        crops = collect_vehicle_crops(tracks, make_frames(n=2))
        assert crops == []

    def test_padding_sinirlara_kirpilir(self):
        # Sol üst köşedeki bbox: padding negatif koordinat üretmez
        tracks = {1: make_track(1, bbox=(0, 0, 20, 20))}
        crops = collect_vehicle_crops(tracks, make_frames(), padding_ratio=0.5)
        assert crops[0]["crop"].shape[:2] == (30, 30)  # (0..20) + sağdan 10 padding


# ----------------------------------------------------------------------
# label_vehicles
# ----------------------------------------------------------------------

class TestLabelVehicles:
    def test_gecerli_yanit_track_haritasi_doner(self):
        tracks = {1: make_track(1), 2: make_track(2)}
        response = json.dumps({"vehicles": [
            {"image_index": 0, "vehicle_type": "forklift", "confidence_hint": "high", "reasoning": "forks visible"},
            {"image_index": 1, "vehicle_type": "truck", "confidence_hint": "medium", "reasoning": "flatbed"},
        ]})
        result = label_vehicles(tracks, make_frames(), MockBackend(response), make_config())
        assert result[1]["vehicle_type"] == "forklift"
        assert result[2]["vehicle_type"] == "truck"

    def test_kapali_kume_disi_tip_atilir(self):
        tracks = {1: make_track(1)}
        response = json.dumps({"vehicles": [
            {"image_index": 0, "vehicle_type": "ufo", "confidence_hint": "high"},
        ]})
        assert label_vehicles(tracks, make_frames(), MockBackend(response), make_config()) == {}

    def test_aralik_disi_image_index_atilir(self):
        tracks = {1: make_track(1)}
        response = json.dumps({"vehicles": [
            {"image_index": 7, "vehicle_type": "forklift", "confidence_hint": "high"},
        ]})
        assert label_vehicles(tracks, make_frames(), MockBackend(response), make_config()) == {}

    def test_bozuk_json_sessiz_fallback(self):
        tracks = {1: make_track(1)}
        assert label_vehicles(tracks, make_frames(), MockBackend("json değil bu"), make_config()) == {}

    def test_vlm_hatasi_sessiz_fallback(self):
        tracks = {1: make_track(1)}
        backend = MockBackend(raise_exc=True)
        assert label_vehicles(tracks, make_frames(), backend, make_config()) == {}

    def test_disabled_noop_ve_vlm_cagrilmaz(self):
        tracks = {1: make_track(1)}
        backend = MockBackend()
        assert label_vehicles(tracks, make_frames(), backend, make_config(enabled=False)) == {}
        assert backend.calls == []

    def test_arac_yoksa_vlm_cagrilmaz(self):
        tracks = {1: make_track(1, class_name="insan")}
        backend = MockBackend()
        assert label_vehicles(tracks, make_frames(), backend, make_config()) == {}
        assert backend.calls == []

    def test_json_fence_temizlenir(self):
        tracks = {1: make_track(1)}
        response = "```json\n" + json.dumps({"vehicles": [
            {"image_index": 0, "vehicle_type": "excavator", "confidence_hint": "low"},
        ]}) + "\n```"
        result = label_vehicles(tracks, make_frames(), MockBackend(response), make_config())
        assert result[1]["vehicle_type"] == "excavator"

    def test_vehicle_types_kapali_kume_tanimli(self):
        assert "forklift" in VEHICLE_TYPES and "other" in VEHICLE_TYPES
        prompt = build_labeling_prompt(3)
        assert "0-2" in prompt  # indeks aralığı prompt'a enjekte ediliyor


# ----------------------------------------------------------------------
# apply_vehicle_labels
# ----------------------------------------------------------------------

def make_observation() -> dict:
    return {
        "frame_idx": 0,
        "detections": [{"class": "arac", "track_id": 1}, {"class": "insan", "track_id": 2}],
        "tracks": [{"class": "arac", "track_id": 1}, {"class": "insan", "track_id": 2}],
        "scene_graph": {"nodes": [
            {"node_id": "arac_1", "class": "arac"},
            {"node_id": "insan_2", "class": "insan"},
        ]},
    }


class TestApplyVehicleLabels:
    def test_track_ve_observation_guncellenir(self):
        tracks = {1: make_track(1)}
        obs = [make_observation()]
        label_map = {1: {"vehicle_type": "forklift", "confidence_hint": "high", "reasoning": ""}}

        updated = apply_vehicle_labels(tracks, obs, label_map)

        assert updated == 1
        assert tracks[1].class_name == "forklift"
        assert tracks[1].last_detection.class_name == "forklift"
        assert obs[0]["detections"][0]["class"] == "forklift"
        assert obs[0]["detections"][1]["class"] == "insan"  # arac olmayan dokunulmaz
        assert obs[0]["tracks"][0]["class"] == "forklift"
        node = obs[0]["scene_graph"]["nodes"][0]
        assert node["class"] == "forklift"
        assert node["node_id"] == "forklift_1"

    def test_bos_harita_noop(self):
        tracks = {1: make_track(1)}
        obs = [make_observation()]
        assert apply_vehicle_labels(tracks, obs, {}) == 0
        assert tracks[1].class_name == "arac"
        assert obs[0]["scene_graph"]["nodes"][0]["node_id"] == "arac_1"

    def test_haritada_olmayan_track_dokunulmaz(self):
        tracks = {1: make_track(1), 5: make_track(5)}
        obs = [make_observation()]
        label_map = {1: {"vehicle_type": "truck", "confidence_hint": "low", "reasoning": ""}}
        apply_vehicle_labels(tracks, obs, label_map)
        assert tracks[5].class_name == "arac"
