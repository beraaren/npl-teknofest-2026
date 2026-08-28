"""UI/backend entegrasyonu için birim testleri.

Bu testler cv2 gibi ağır bağımlılıkları içermez; gateway store ve yeni
RAGLayer öneri eşleştirmesini doğrular.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile

from src.reasoning.rag_layer import RAGLayer
from backend.gateway import store


def test_rag_layer_suggestions_empty_query():
    """Boş sorguda tüm öneriler önceliğe göre döner."""
    rag = RAGLayer()
    results = rag.match_suggestions(event_types=[], query_text="", top_k=10)
    assert len(results) == 4
    ids = {r["oneri_id"] for r in results}
    assert ids == {"forklift_yaya_ayrisi", "kkd_takip_sistemi",
                   "kaygan_zemin_protokolu", "forklift_hiz_limiti"}


def test_rag_layer_suggestions_pattern_match():
    """event_types related_patterns ile eşleştiğinde ilgili öneri öne çıkar."""
    rag = RAGLayer()
    results = rag.match_suggestions(event_types=["ppe_missing"], query_text="baret")
    assert results
    assert results[0]["oneri_id"] == "kkd_takip_sistemi"
    assert results[0]["pattern_eslesmesi"] is True
    assert "maliyet_tahmini" in results[0]


def test_rag_layer_get_suggestion():
    """get_suggestion tek kayıt döner; yoksa None."""
    rag = RAGLayer()
    s = rag.get_suggestion("kkd_takip_sistemi")
    assert s is not None
    assert s["baslik"].startswith("KKD")
    assert rag.get_suggestion("olmayan_id") is None


def test_store_feedback_crud(tmp_path):
    """RLHF feedback CRUD ve istatistik döngüsü çalışır."""
    original_db = store.DB_PATH
    db_path = tmp_path / "test_feedback_gateway.db"
    store.DB_PATH = str(db_path)
    try:
        store.init_db()

        # 1. Pozitif geri bildirim (doğru karar)
        fb1 = store.create_feedback(
            analysis_slug="analysis_01",
            camera_id="cam-01",
            feedback_type="correct",
            original_risk="Yüksek",
            original_summary="Forklift devrilme tehlikesi",
            original_output={"risk": "Yüksek", "summary": "Forklift devrilme tehlikesi"},
            supervisor_notes="Doğrulandı.",
        )
        assert fb1["id"] >= 1
        assert fb1["feedback_type"] == "correct"
        assert fb1["original_risk"] == "Yüksek"

        # 2. Negatif geri bildirim (yanlış alarm / düzeltme)
        fb2 = store.create_feedback(
            analysis_slug="analysis_02",
            camera_id="cam-02",
            feedback_type="false_positive",
            original_risk="Yüksek",
            original_summary="Tehlikeli yakınlaşma",
            original_output={"risk": "Yüksek", "summary": "Tehlikeli yakınlaşma"},
            corrected_risk="Düşük",
            corrected_summary="Personel güvenli yaya yolunda ilerliyor",
            supervisor_notes="Yanlış alarm, yaya yolu güvenli.",
        )
        assert fb2["id"] >= 2
        assert fb2["feedback_type"] == "false_positive"
        assert fb2["corrected_risk"] == "Düşük"

        # Listeleme
        all_feedbacks = store.get_feedbacks()
        assert len(all_feedbacks) == 2

        by_slug = store.get_feedback_by_slug("analysis_02")
        assert by_slug is not None
        assert by_slug["corrected_risk"] == "Düşük"

        # İstatistikler
        stats = store.get_feedback_stats()
        assert stats["total"] == 2
        assert stats["correct_count"] == 1
        assert stats["correction_count"] == 1
        assert stats["accuracy_rate"] == 50.0

        # DPO export
        records = store.export_dpo_dataset_records()
        assert len(records) == 2
        assert "chosen" in records[0]
        assert "rejected" in records[0]
        assert "prompt" in records[0]

        jsonl = store.export_dpo_dataset_jsonl()
        lines = [l for l in jsonl.strip().split("\n") if l]
        assert len(lines) == 2

        # Sadece düzeltmeler export
        corr_records = store.export_dpo_dataset_records(only_corrections=True)
        assert len(corr_records) == 1
        assert corr_records[0]["metadata"]["feedback_type"] == "false_positive"

    finally:
        store.DB_PATH = original_db



def test_rag_layer_common_conditions_has_ten_costed_items():
    """Hızlı seçim katalogdan on yaygın, maliyetli durum alır."""
    conditions = RAGLayer().common_conditions()
    assert len(conditions) == 10
    assert {item["condition_id"] for item in conditions} >= {
        "forklift_yaya_yakinligi",
        "kkd_eksikligi",
        "yangin_veya_duman",
    }
    for item in conditions:
        cost = item["maliyet_tahmini"]
        assert cost["alt_sinir_tl"] > 0
        assert cost["ust_sinir_tl"] >= cost["alt_sinir_tl"]
        assert cost["para_birimi"] == "TRY"


def test_suggestion_query_reports_known_and_custom_condition_costs():
    """Katalog seçimi maliyet taşır; serbest metin kabul edilir ve maliyeti bilinmez."""
    from backend.gateway.routers.ops import SuggestionQuery, query_suggestions

    known = asyncio.run(query_suggestions(SuggestionQuery(
        condition_id="kkd_eksikligi",
        query_text="KKD eksikliği (baret/yelek)",
    )))
    assert known["selected_condition"]["cost_known"] is True
    assert known["selected_condition"]["maliyet_tahmini"]["alt_sinir_tl"] == 8000

    custom = asyncio.run(query_suggestions(SuggestionQuery(query_text="Özel havalandırma sorunu")))
    assert custom["selected_condition"] == {
        "condition_id": None,
        "baslik": "Özel havalandırma sorunu",
        "maliyet_tahmini": None,
        "cost_known": False,
    }
