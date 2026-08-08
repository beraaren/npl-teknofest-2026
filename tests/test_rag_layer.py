"""RAGLayer vektörel arama testleri (plan/04)."""
from pathlib import Path

from src.reasoning.rag_layer import RAGLayer

DATA = Path(__file__).resolve().parent.parent / "data"


def make_rag() -> RAGLayer:
    return RAGLayer(
        patterns_path=DATA / "risk_patterns.yaml",
        actions_path=DATA / "action_catalog.yaml",
    )


def test_signal_boosts_matching_pattern():
    """dangerous_proximity sinyali ilgili pattern'i doğrudan işaretler (isim uyumu düzeltildi)."""
    rag = make_rag()
    ctx = rag.build_context("sıradan bir saha görüntüsü", [
        {"event_type": "dangerous_proximity", "timestamp": "00:05", "description": "", "confidence": 0.9,
         "involved_track_ids": [1, 2], "metadata": {}},
    ])
    names = [m["pattern"] for m in ctx["matched_patterns"]]
    assert "dangerous_proximity" in names
    assert ctx["risk_level"] == "Yüksek"
    assert ctx["risk_score"] == 80


def test_empty_inputs_return_safe_default():
    rag = make_rag()
    ctx = rag.build_context("", [])
    assert ctx == {"risk_level": "Düşük", "risk_score": 0, "actions": [], "matched_patterns": []}


def test_vector_search_matches_smoke_report():
    """Ana sorgu (Observer raporu) 'duman' içerdiğinde fire_smoke pattern'i vektör benzerliğiyle eşleşir."""
    rag = make_rag()
    ctx = rag.build_context("sahada yoğun duman görülüyor", [])
    names = [m["pattern"] for m in ctx["matched_patterns"]]
    assert "fire_smoke" in names
    smoke = next(m for m in ctx["matched_patterns"] if m["pattern"] == "fire_smoke")
    assert smoke["similarity"] >= 0.1
    assert ctx["risk_level"] == "Yüksek"


def test_output_contract_keys():
    rag = make_rag()
    ctx = rag.build_context("forklift devrilmiş olabilir", [
        {"event_type": "forklift_tip_over", "timestamp": "00:02", "description": "", "confidence": 0.9,
         "involved_track_ids": [5], "metadata": {}},
    ])
    assert set(ctx.keys()) == {"risk_level", "risk_score", "actions", "matched_patterns"}
    assert isinstance(ctx["actions"], list)
    assert ctx["risk_score"] == 95  # tip_over en yüksek skorlu eşleşme
