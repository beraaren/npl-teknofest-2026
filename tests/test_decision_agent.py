"""DecisionAgent (VLM karar ajanı) testleri — sahte backend ile."""
import json
from pathlib import Path

from src.config import DecisionAgentConfig, GuardrailConfig, VLMConfig
from src.output.guardrail import OutputGuardrail
from src.output.schema import AnalysisOutput
from src.reasoning.decision_agent import DecisionAgent
from src.reasoning.mock_tools import MockToolRegistry
from src.reasoning.rag_layer import RAGLayer

DATA = Path(__file__).resolve().parent.parent / "data"


class FakeBackend:
    """VLMBackend yerine geçer; çağrıları kaydeder, hazır JSON döner."""

    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def name(self) -> str:
        return "vllm"

    def generate(self, images, prompt, temperature=0.15, max_tokens=1024) -> str:
        self.calls.append({"prompt": prompt, "temperature": temperature})
        return self.response


CANNED_DECISION = json.dumps({
    "summary": "Forklift devrilmesi tespit edildi.",
    "events": [{"time": "00:03", "event": "Forklift devrildi", "event_type": "forklift_tip_over", "confidence": 0.9}],
    "risk": "Yüksek",
    "actions": ["Alanı güvenliğe al", "Sağlık ekibini çağır"],
    "reasoning": "Kural motoru üç kare üst üste devrilme bildirdi; VLM de doğruladı.",
    "confidence": 0.9,
    "triggered_mock_tools": [{"tool_name": "stop_forklift", "params": {"location": "saha"}}],
}, ensure_ascii=False)


def make_agent(response=CANNED_DECISION):
    backend = FakeBackend(response)
    tools = MockToolRegistry(tools_path=DATA / "mock_tools.yaml")
    agent = DecisionAgent(
        config=DecisionAgentConfig(system_prompt="TEST SISTEM PROMPTU: kanıtları tart."),
        vlm_config=VLMConfig(),
        rag=None,
        memory=None,
        tools=tools,
        backend=backend,
    )
    return agent, backend


def sample_inputs():
    signals = [{
        "event_type": "forklift_tip_over", "timestamp": "00:03",
        "description": "Forklift devrilmesi tespit edildi", "confidence": 0.95,
        "involved_track_ids": [2], "metadata": {"aspect_ratio": 1.8},
    }]
    graphs = [{"frame_idx": 0, "timestamp": 3.0, "nodes": [], "edges": []}]
    rag_ctx = {"hypotheses": [], "unverified_hypotheses": [], "recommended_actions": [], "matched_patterns": []}
    vlm = {"scene_summary_tr": "devrilmiş araç", "detected_entities": [], "detected_actions_tr": [],
           "risk_flags_tr": ["araç devrilmesi"], "confidence_overall": 0.8, "notable_frames": [1]}
    return signals, graphs, rag_ctx, vlm


def test_decide_contract_and_guardrail_passthrough():
    """decide() guardrail hazır raw_text döner; sahte VLM çıktısı şemadan geçer."""
    agent, _ = make_agent()
    signals, graphs, rag_ctx, vlm = sample_inputs()
    raw = agent.decide(event_signals=signals, scene_graphs=graphs, rag_context=rag_ctx, vlm_interpretation=vlm)

    assert set(raw.keys()) == {"raw_text", "retry_fn", "scene_context", "candidate_observations"}
    assert raw["candidate_observations"][0]["event_type"] == "forklift_tip_over"

    out = OutputGuardrail(GuardrailConfig()).validate(raw["raw_text"], raw["retry_fn"])
    assert out["summary"] != "Bilmiyorum"
    assert out["risk"] == "Yüksek"


def test_prompt_feeds_all_evidence_channels():
    """Üç kanal + RAG + kapalı araç kümesi prompt'a girer; düşünce akışı system prompt'tan gelir."""
    agent, backend = make_agent()
    signals, graphs, rag_ctx, vlm = sample_inputs()
    agent.decide(event_signals=signals, scene_graphs=graphs, rag_context=rag_ctx, vlm_interpretation=vlm)

    prompt = backend.calls[0]["prompt"]
    assert "TEST SISTEM PROMPTU" in prompt            # düşünce akışı config'den besleniyor
    assert "forklift_tip_over" in prompt              # S3 olay sinyalleri
    assert "araç devrilmesi" in prompt                # S8 VLM yorumu
    assert "matched_patterns" in prompt               # S7 RAG konteksti
    assert "stop_forklift" in prompt                  # kapalı araç kümesi
    assert "reasoning" in prompt                      # nihai şema talimatı


def test_retry_fn_regenerates_with_lower_temperature():
    """Guardrail retry'ı aynı prompt'u düşük sıcaklıkla yeniden çağırır."""
    agent, backend = make_agent()
    signals, graphs, rag_ctx, vlm = sample_inputs()
    raw = agent.decide(event_signals=signals, scene_graphs=graphs, rag_context=rag_ctx, vlm_interpretation=vlm)
    raw["retry_fn"](0.05)
    assert backend.calls[-1]["temperature"] == 0.05
    assert len(backend.calls) == 2


def test_interpret_frames_parses_structured_json():
    """S8: VLM yapılandırılmış JSON dönerse alanlar aynen alınır."""
    response = json.dumps({
        "scene_summary_tr": "sisli saha", "detected_entities": [],
        "detected_actions_tr": ["yürüme"], "risk_flags_tr": ["duman"],
        "confidence_overall": 0.7, "notable_frames": [0],
    })
    agent, _ = make_agent(response=response)
    import numpy as np
    out = agent.interpret_frames([np.zeros((4, 4, 3), dtype=np.uint8)])
    assert out["risk_flags_tr"] == ["duman"]
    assert out["confidence_overall"] == 0.7


def test_interpret_frames_fallback_on_free_text():
    """S8: VLM serbest metin dönerse özet olarak saklanır, pipeline kırılmaz."""
    agent, _ = make_agent(response="Sahada duman var gibi görünüyor.")
    import numpy as np
    out = agent.interpret_frames([np.zeros((4, 4, 3), dtype=np.uint8)])
    assert "duman" in out["scene_summary_tr"]
    assert out["risk_flags_tr"] == []


def test_garbage_vlm_output_falls_to_guardrail_null_response():
    """VLM geçersiz çıktı verirse guardrail retry sonrası 'Bilmiyorum'a düşer (KPI: null-response)."""
    agent, _ = make_agent(response="bu bir json değil")
    signals, graphs, rag_ctx, vlm = sample_inputs()
    raw = agent.decide(event_signals=signals, scene_graphs=graphs, rag_context=rag_ctx, vlm_interpretation=vlm)
    out = OutputGuardrail(GuardrailConfig()).validate(raw["raw_text"], raw["retry_fn"])
    assert out["summary"] == "Bilmiyorum"
    validated = AnalysisOutput.model_validate(out)
    assert validated.risk == "Düşük"
