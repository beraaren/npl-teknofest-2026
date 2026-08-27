"""RAGLayer'ın hipotez üretim sözleşmesi testleri."""
from pathlib import Path

from src.reasoning.rag_layer import RAGLayer

DATA = Path(__file__).resolve().parent.parent / "data"


def make_rag() -> RAGLayer:
    return RAGLayer(
        patterns_path=DATA / "risk_patterns.yaml",
        actions_path=DATA / "action_catalog.yaml",
    )


def test_signal_creates_candidate_hypothesis_without_risk_score():
    rag = make_rag()
    context = rag.build_context("sıradan bir saha görüntüsü", [
        {"event_type": "dangerous_proximity", "timestamp": "00:05", "description": "", "confidence": 0.9,
         "involved_track_ids": [1, 2], "metadata": {}},
    ])
    names = [item["pattern"] for item in context["hypotheses"]]
    assert "dangerous_proximity" in names
    candidate = next(item for item in context["hypotheses"] if item["pattern"] == "dangerous_proximity")
    assert candidate["evidence_status"] == "signal_candidate"
    assert "risk_level" not in candidate
    assert "risk_score" not in candidate


def test_empty_inputs_return_empty_hypothesis_contract():
    rag = make_rag()
    context = rag.build_context("", [])
    assert context["hypotheses"] == []
    assert context["unverified_hypotheses"] == []
    assert context["recommended_actions"] == []
    assert context["matched_patterns"] == []


def test_vector_search_is_unverified_hypothesis_not_risk_decision():
    rag = make_rag()
    context = rag.build_context("sahada yoğun duman görülüyor", [])
    names = [item["pattern"] for item in context["unverified_hypotheses"]]
    assert "fire_smoke" in names
    smoke = next(item for item in context["unverified_hypotheses"] if item["pattern"] == "fire_smoke")
    assert smoke["evidence_status"] == "unverified"
    assert smoke["similarity"] > 0


def test_gathering_structural_match_requires_three_people():
    rag = make_rag()
    one_person = [{"detections": [{"class": "insan"}], "tracks": []}]
    three_people = [{"detections": [{"class": "insan"}, {"class": "insan"}, {"class": "insan"}], "tracks": []}]
    assert "gathering" not in rag.match_patterns([], one_person)
    assert "gathering" in rag.match_patterns([], three_people)
