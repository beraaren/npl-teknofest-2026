"""Guardrail testleri."""
import pytest

from src.config import GuardrailConfig
from src.output.guardrail import OutputGuardrail


def test_guardrail_accepts_valid_json():
    raw = '''
    ```json
    {
        "summary": "Test özet",
        "events": [{"time": "00:05", "event": "x", "event_type": "fall", "confidence": 0.8}],
        "risk": "Yüksek",
        "actions": ["a1", "a2"],
        "reasoning": "r",
        "confidence": 0.8,
        "triggered_mock_tools": []
    }
    ```
    '''
    guardrail = OutputGuardrail(GuardrailConfig())
    result = guardrail.validate(raw, lambda t: raw, rag_risk_level="Yüksek")
    assert result["risk"] == "Yüksek"
    assert len(result["actions"]) == 2


def test_guardrail_retries_and_null_response():
    bad = "Bu geçerli bir JSON değil"
    guardrail = OutputGuardrail(GuardrailConfig(max_retries=2, temperatures=[0.1, 0.05]))
    result = guardrail.validate(bad, lambda t: bad, rag_risk_level="Düşük")
    assert result["summary"] == "Bilmiyorum"


def test_guardrail_normalizes_malformed_events():
    """Demoda görülen gerçek hata: aralıklı/sıfırsız time + eksik confidence kurtarılmalı."""
    raw = '''
    {
        "summary": "Araç devrilmesi",
        "events": [
            {"time": "0:01-0:04", "event": "araç devrildi", "event_type": "tip_over"},
            {"time": "0:05-0:07", "event": "kişiler toplandı", "event_type": "gathering", "confidence": 0.9}
        ],
        "risk": "Yüksek",
        "actions": ["alanı güvene al", "sağlık ekibi çağır"],
        "reasoning": "r",
        "confidence": 0.8,
        "triggered_mock_tools": []
    }
    '''
    guardrail = OutputGuardrail(GuardrailConfig())
    result = guardrail.validate(raw, lambda t: raw, rag_risk_level="Yüksek")
    assert result["summary"] == "Araç devrilmesi"
    assert result["events"][0]["time"] == "00:01"
    assert result["events"][0]["confidence"] == 0.5  # varsayılan
    assert result["events"][1]["time"] == "00:05"
    assert result["events"][1]["confidence"] == 0.9


def test_guardrail_normalizes_string_mock_tool_calls():
    raw = '''
    {
        "summary": "Doğrulanmış olay",
        "events": [],
        "risk": "Düşük",
        "actions": [],
        "reasoning": "r",
        "confidence": 0.8,
        "triggered_mock_tools": [
            "secure_area",
            {"tool_name": "notify_supervisor", "params": {"priority": "high"}}
        ]
    }
    '''
    guardrail = OutputGuardrail(GuardrailConfig())
    result = guardrail.validate(raw, lambda t: raw)

    assert result["triggered_mock_tools"] == [
        {"tool_name": "secure_area", "params": {}},
        {"tool_name": "notify_supervisor", "params": {"priority": "high"}},
    ]


def test_guardrail_accepts_canonical_high_result_and_top_level_tool_string():
    raw = '''
    {
        "summary": "Hareketli makine ile yaya arasında ezilme riski var.",
        "results": [{
            "result_id": "result-1",
            "result_type": "incident",
            "time": "00:10",
            "timestamp_sec": 10,
            "event": "Yaya, hareketli makinenin çalışma alanında.",
            "event_type": "dangerous_proximity",
            "hazard_mechanism": "Makinenin çarpması veya ezmesi",
            "severity": "high",
            "evidence": {"agreement": "corroborated"}
        }],
        "risk": "Yüksek",
        "overall_risk": "high",
        "actions": ["Makineyi durdur ve alanı ayır."],
        "reasoning": "Geometrik ve görsel kanıtlar aynı tehlikeyi destekliyor.",
        "triggered_mock_tools": "secure_area"
    }
    '''
    guardrail = OutputGuardrail(GuardrailConfig())
    result = guardrail.validate(raw, lambda t: raw)

    assert result["summary"] != "Bilmiyorum"
    assert result["overall_risk"] == "high"
    assert result["results"][0]["severity"] == "high"
    assert result["triggered_mock_tools"] == [{"tool_name": "secure_area", "params": {}}]


def test_normalize_time_variants():
    assert OutputGuardrail._normalize_time("0:01-0:04") == "00:01"
    assert OutputGuardrail._normalize_time("12:34") == "12:34"
    assert OutputGuardrail._normalize_time("1:02:03") == "62:03"
    assert OutputGuardrail._normalize_time(5.0) == "5.0"
