"""UI/backend entegrasyonu için birim testleri.

Bu testler cv2 gibi ağır bağımlılıkları içermez; gateway store ve yeni
RAGLayer öneri eşleştirmesini doğrular.
"""
from __future__ import annotations

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


def test_store_field_alerts(tmp_path):
    """field_alerts CRUD döngüsü çalışır."""
    # Her test izole DB kullansın
    original_db = store.DB_PATH
    db_path = tmp_path / "test_gateway.db"
    store.DB_PATH = str(db_path)
    try:
        store.init_db()
        row = store.create_field_alert(
            camera_id="cam-01",
            risk="Yüksek",
            headline="Düşme alarmı",
            summary="Test summary",
            actions=["Aksiyon A"],
            risk_segment={"start_sec": 2, "end_sec": 6, "event_type": "person_fall"},
            target_roles=["sağlık"],
        )
        assert row["id"] >= 1
        assert row["camera_id"] == "cam-01"

        all_alerts = store.get_field_alerts()
        assert len(all_alerts) == 1

        # rol filtresi doğru çalışmalı
        filtered = store.get_field_alerts(role="sağlık")
        assert len(filtered) == 1
        assert filtered[0]["target_roles"] == ["sağlık"]

        filtered_no_match = store.get_field_alerts(role="teknisyen")
        assert len(filtered_no_match) == 0
    finally:
        store.DB_PATH = original_db


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

