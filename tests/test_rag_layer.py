"""RAGLayer'ın hipotez üretim sözleşmesi testleri."""
from pathlib import Path

from src.reasoning.rag_layer import RAGLayer, _vlm_interpretation_to_query

DATA = Path(__file__).resolve().parent.parent / "data"


def make_rag() -> RAGLayer:
    return RAGLayer(
        patterns_path=DATA / "risk_patterns.yaml",
        actions_path=DATA / "action_catalog.yaml",
    )


def test_signal_creates_candidate_hypothesis_with_catalog_risk_info():
    rag = make_rag()
    context = rag.build_context("sıradan bir saha görüntüsü", [
        {"event_type": "dangerous_proximity", "timestamp": "00:05", "description": "", "confidence": 0.9,
         "involved_track_ids": [1, 2], "metadata": {}},
    ])
    names = [item["pattern"] for item in context["hypotheses"]]
    assert "dangerous_proximity" in names
    candidate = next(item for item in context["hypotheses"] if item["pattern"] == "dangerous_proximity")
    assert candidate["evidence_status"] == "signal_candidate"
    # risk_score/risk_level RAG’ın kendi kararı değil, katalog referansıdır.
    assert candidate["risk_level"] == "Yüksek"
    assert candidate["risk_score"] == 80
    assert "indicators" in candidate and candidate["indicators"]
    assert "required_nodes" in candidate and candidate["required_nodes"]
    # retrieval_confidence vektör benzerliğidir; 0-1 aralığında ve amaç etiketi taşır.
    assert 0.0 <= candidate["retrieval_confidence"] <= 1.0
    assert "hypothesis_purpose" in candidate
    assert "post_incident_response" in candidate["hypothesis_purpose"]
    assert "pre_incident_risk" in candidate["hypothesis_purpose"]


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


def test_vlm_interpretation_to_query_includes_all_relevant_fields():
    vlm = {
        "scene_summary_tr": "Forklift yüksek rafa çarptı ve devrildi.",
        "detected_actions_tr": ["Sürücü araçtan atladı.", "Raf yere düştü."],
        "detected_entities": [
            {"label": "forklift", "notes_tr": "yeşil forklift, üzerinde 73 numarası"},
            {"label": "rack", "notes_tr": "yüksek raf yapısı"},
        ],
        "risk_events": [
            {"description_tr": "Yüksek rafa çarpma ve devrilme riski.", "severity": "critical"}
        ],
        "risk_flags_tr": ["yüksek raf çarpışması", "forklift devrilme"],
    }
    query = _vlm_interpretation_to_query(vlm)
    assert "Forklift yüksek rafa çarptı ve devrildi." in query
    assert "Sürücü araçtan atladı." in query
    assert "Raf yere düştü." in query
    assert "forklift" in query
    assert "yeşil forklift, üzerinde 73 numarası" in query
    assert "Yüksek rafa çarpma ve devrilme riski." in query
    assert "critical" in query
    assert "yüksek raf çarpışması" in query


def test_vlm_interpretation_boosts_forklift_tip_over_similarity():
    rag = make_rag()
    vlm = {
        "scene_summary_tr": "Forklift yüksek rafa çarptı ve devrildi.",
        "detected_actions_tr": ["Sürücü araçtan atladı."],
        "detected_entities": [{"label": "forklift"}],
        "risk_events": [{"description_tr": "Yüksek rafa çarpma ve devrilme riski.", "severity": "critical"}],
        "risk_flags_tr": [],
    }
    without_vlm = rag.build_context("", [])
    with_vlm = rag.build_context("", [], vlm_interpretation=vlm)

    def top_pattern(context):
        return context["unverified_hypotheses"][0]["pattern"] if context["unverified_hypotheses"] else None

    def similarity_of(context, pattern):
        for item in context["unverified_hypotheses"]:
            if item["pattern"] == pattern:
                return item["similarity"]
        return 0.0

    # VLM yorumu eklenince forklift_tip_over benzerliği artmalı.
    assert similarity_of(with_vlm, "forklift_tip_over") >= similarity_of(without_vlm, "forklift_tip_over")


def test_unverified_hypothesis_has_retrieval_confidence_and_purpose():
    rag = make_rag()
    context = rag.build_context("sahada yoğun duman görülüyor", [])
    assert context["unverified_hypotheses"]
    smoke = next(item for item in context["unverified_hypotheses"] if item["pattern"] == "fire_smoke")
    assert smoke["evidence_status"] == "unverified"
    assert 0.0 <= smoke["retrieval_confidence"] <= 1.0
    assert smoke["retrieval_confidence"] > 0
    assert "hypothesis_purpose" in smoke
    # fire_smoke hem risk göstergesi hem de yangın söndürme/araç ipuçları içerir.
    assert "pre_incident_risk" in smoke["hypothesis_purpose"]
    assert "post_incident_response" in smoke["hypothesis_purpose"]


def test_retrieval_confidence_is_similarity_clipped():
    rag = make_rag()
    context = rag.build_context("sıradan bir saha görüntüsü", [
        {"event_type": "dangerous_proximity", "timestamp": "00:05", "description": "", "confidence": 0.9,
         "involved_track_ids": [1, 2], "metadata": {}},
    ])
    candidate = next(item for item in context["hypotheses"] if item["pattern"] == "dangerous_proximity")
    # boost sonrası similarity 1.0'ı aşabilir; retrieval_confidence kırpılmış olmalı.
    assert candidate["retrieval_confidence"] <= 1.0
