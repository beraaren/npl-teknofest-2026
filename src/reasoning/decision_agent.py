"""Karar Ajanı: üç kanaldan gelen kanıtı VLM düşünce zinciriyle birleştirir.

Canvas'taki birleşim noktası: Observer raporu (S6) + RAG konteksti (S7) +
VLM yorumu (S8) burada buluşur. Kanıtlar modele yapılandırılmış olarak verilir;
çelişki çözümü, güven değerlendirmesi ve nihai format karar ajanının (VLM)
kendi muhakemesine aittir — skor formülü yoktur. Düşünce akışı system
prompt'unda tanımlıdır (config.yaml -> decision_agent.system_prompt).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import numpy as np
from numpy.typing import NDArray

from ..config import DecisionAgentConfig, VLMConfig
# Doğrudan içe aktarım bilinçlidir. Burada eskiden bir try/except zinciri vardı
# ve son dalda `create_backend = None` atanıyordu; modül bulunamadığında hata
# içe aktarma anında değil, ilk çağrıda "'NoneType' object is not callable"
# olarak ortaya çıkıyordu. Bu, mikroservis tarafında sessizce yutulup "risk:
# Düşük" üreten bir arızaya dönüşmüştü. Eksik bağımlılık artık hemen ve
# anlaşılır biçimde bildirilir.
from ..models.vlm_backend import VLMBackend, create_backend
from ..utils.logger import get_logger
from .context_builder import build_candidate_observations, build_scene_context
from .memory import ShortTermMemory
from .mock_tools import MockToolRegistry
from .rag_layer import RAGLayer


# Kanal B prompt'u: VLM'den yapılandırılmış JSON ister (S8 sözleşmesi).
# Spesifik sınıf adları (forklift, palet...) bilinçli istenmez — genel terimlerle
# doğruluk artar; spesifik tanımları algı katmanı (YOLO) sağlar.
STRUCTURED_OBSERVATION_PROMPT = (
    "These images are video frames from a work site or surveillance camera. "
    "Describe what you see carefully.\n\n"
    "VEHICLE IDENTIFICATION — THINK STEP BY STEP:\n"
    "When you see ANY vehicle or machine, reason about its exact type before labeling it. "
    "Consider visual cues: size, shape, wheels vs tracks, cabin position, forks, boom arm, "
    "bucket, flatbed, road context vs industrial site. Possible types include: forklift, "
    "crane, excavator, loader, truck, pickup, car, van, bus, motorcycle, bicycle. "
    "A vehicle on a public road is most likely a car/truck/bus — do NOT assume industrial equipment. "
    "Write your reasoning in \"vehicle_type_reasoning\".\n\n"
    "For non-vehicle objects use general terms: 'person/human', 'load/object', "
    "'rack/structure', 'liquid/substance', 'smoke or flame'.\n\n"
    "Give your answer ONLY in accordance with the following JSON schema, do not add explanations:\n"
    "{\n"
    '  "scene_summary_tr": "scene summary",\n'
    '  "vehicle_type_reasoning": "step-by-step reasoning about what type each vehicle/machine is and why",\n'
    '  "detected_entities": [{"label": "specific vehicle type or object label", "confidence_hint": "low|mid|high", "notes_tr": "note"}],\n'
    '  "detected_actions_tr": ["observed actions"],\n'
    '  "risk_flags_tr": ["risky situations that stand out: tip-over, fall, smoke, leakage, gathering..."],\n'
    '  "confidence_overall": 0.0-1.0,\n'
    '  "notable_frames": [0]\n'
    "}"
)


def _extract_json(text: str) -> Dict[str, Any] | None:
    """```json fence'lerini temizleyip ilk { ... } bloğunu parse etmeyi dener."""
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


class DecisionAgent:
    """VLM tabanlı karar ajanı; kanıtları birleştirip nihai kararı kendi yazar."""

    def __init__(
        self,
        config: DecisionAgentConfig,
        vlm_config: VLMConfig,
        rag: RAGLayer,
        memory: ShortTermMemory,
        tools: MockToolRegistry,
        backend: VLMBackend | None = None,
    ):
        self.config = config
        self.vlm_config = vlm_config
        self.rag = rag
        self.memory = memory
        self.tools = tools
        self.backend = backend
        self.logger = get_logger("DecisionAgent")

    def _ensure_backend(self) -> VLMBackend:
        if self.backend is None:
            self.backend = create_backend(self.vlm_config)
        return self.backend

    # ------------------------------------------------------------------
    # Kanal B: VLM yorumu (S8 — yapılandırılmış JSON)
    # ------------------------------------------------------------------
    def interpret_frames(
        self,
        images: List[NDArray[np.uint8]],
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """Kritik kareleri bağımsız yorumlar; yapılandırılmış vlm_interpretation döner."""
        if not images:
            return self._empty_vlm()
        backend = self._ensure_backend()
        temperature = self.vlm_config.vllm.temperature
        if backend.name() == "llama_cpp":
            temperature = self.vlm_config.llama_cpp.temperature
        raw = backend.generate(
            images, STRUCTURED_OBSERVATION_PROMPT, temperature=temperature, max_tokens=max_tokens
        )
        self.logger.debug(f"Kanal B VLM yorumu:\n{raw}")

        parsed = _extract_json(raw)
        if not parsed:
            # Parse başarısız: ham metni özet yap, risk bayrağı yok sayılır
            out = self._empty_vlm()
            out["scene_summary_tr"] = raw.strip()
            return out
        out = self._empty_vlm()
        out.update({k: v for k, v in parsed.items() if k in out})
        if not isinstance(out["risk_flags_tr"], list):
            out["risk_flags_tr"] = []
        if not isinstance(out["detected_actions_tr"], list):
            out["detected_actions_tr"] = []
        try:
            out["confidence_overall"] = float(out["confidence_overall"])
        except (TypeError, ValueError):
            out["confidence_overall"] = 0.5
        return out

    @staticmethod
    def _empty_vlm() -> Dict[str, Any]:
        return {
            "scene_summary_tr": "",
            "detected_entities": [],
            "detected_actions_tr": [],
            "risk_flags_tr": [],
            "confidence_overall": 0.5,
            "notable_frames": [],
        }

    # ------------------------------------------------------------------
    # Karar: kanıtları topla, VLM'in düşünce zinciriyle nihai kararı yazdır
    # ------------------------------------------------------------------
    def decide(
        self,
        event_signals: List[Dict[str, Any]],
        scene_graphs: List[Dict[str, Any]],
        rag_context: Dict[str, Any],
        vlm_interpretation: Dict[str, Any] | None = None,
        images: List[NDArray[np.uint8]] | None = None,
    ) -> Dict[str, Any]:
        """Kanıt paketini VLM'e verir; guardrail hazır decision_raw döner."""
        backend = self._ensure_backend()
        vlm = vlm_interpretation or self._empty_vlm()
        scene_context = build_scene_context(scene_graphs, vlm)
        candidate_observations = build_candidate_observations(event_signals, vlm)
        prompt = self._build_prompt(
            event_signals, scene_graphs, rag_context, vlm,
            scene_context=scene_context,
            candidate_observations=candidate_observations,
        )
        self.logger.debug(f"Karar promptu uzunluğu: {len(prompt)} karakter")

        temperature, max_tokens = self._generation_params(backend)
        raw = backend.generate(images or [], prompt, temperature=temperature, max_tokens=max_tokens)
        self.logger.debug(f"Karar ajanı çıktısı:\n{raw}")

        def retry_fn(temp: float) -> str:
            return backend.generate(images or [], prompt, temperature=temp, max_tokens=max_tokens)

        return {
            "raw_text": raw,
            "retry_fn": retry_fn,
            "scene_context": scene_context,
            "candidate_observations": candidate_observations,
        }

    def _generation_params(self, backend: VLMBackend) -> tuple[float, int]:
        name = backend.name()
        if name == "server":
            return self.vlm_config.server.temperature, self.vlm_config.server.max_tokens
        if name == "llama_cpp":
            return self.vlm_config.llama_cpp.temperature, self.vlm_config.llama_cpp.max_tokens
        if name == "transformers":
            return 0.15, self.vlm_config.transformers.max_new_tokens
        return self.vlm_config.vllm.temperature, self.vlm_config.vllm.max_new_tokens

    # ------------------------------------------------------------------
    # Prompt: kanıt paketi + düşünce akışı (system prompt) + çıktı şeması
    # ------------------------------------------------------------------
    def _build_prompt(
        self,
        event_signals: List[Dict[str, Any]],
        scene_graphs: List[Dict[str, Any]],
        rag_context: Dict[str, Any],
        vlm: Dict[str, Any],
        scene_context: Dict[str, Any] | None = None,
        candidate_observations: List[Dict[str, Any]] | None = None,
    ) -> str:
        parts = [self.config.system_prompt]

        parts.append("\n--- SAHNE / FAALİYET BAĞLAMI (gözlenebilir veriler) ---")
        parts.append(json.dumps(scene_context or build_scene_context(scene_graphs, vlm), ensure_ascii=False, indent=2))

        parts.append("\n--- KÜMELENMİŞ ADAY GÖZLEMLER (risk kararı değildir) ---")
        candidates = candidate_observations or build_candidate_observations(event_signals, vlm)
        parts.append(json.dumps(candidates, ensure_ascii=False, indent=2) if candidates else "Aday geometrik gözlem yok.")

        parts.append("\n--- RAG TEHLİKE HİPOTEZLERİ (kanıt değil) ---")
        parts.append(json.dumps(rag_context, ensure_ascii=False, indent=2))

        parts.append("\n--- VLM GÖRSEL YORUMU (bağımsız kanal) ---")
        parts.append(json.dumps(vlm, ensure_ascii=False, indent=2))

        if self.config.include_scene_graph and scene_graphs:
            parts.append("\n--- SAHNE GRAFİ (SON KARE) ---")
            parts.append(json.dumps(scene_graphs[-1], ensure_ascii=False, indent=2))

        if self.memory is not None:
            parts.append("\n--- KISA SÜRELİ HAFIZA ---")
            parts.append(self.memory.to_prompt_context())

        parts.append("\n--- KULLANILABİLİR ARAÇLAR (kapalı küme — bunların dışına çıkma) ---")
        parts.append(self._tool_catalog_text())

        parts.append(
            "\nZAMAN ALANLARI: 'time' başlangıç MM:SS, 'timestamp_sec' videonun başından "
            "itibaren saniye, 'duration' gözlenen sonuç süresidir. Zamanı bilmiyorsan "
            "0 kullan ve uncertainty_reason içinde nedenini yaz.\n"
            "\nKanıtları bağlam içinde değerlendir ve SADECE aşağıdaki JSON şemasına uygun "
            "Türkçe yanıt ver. 'results' kanoniktir: RAG hipotezi tek başına kanıt değildir; "
            "overall_risk hiçbir result severity'sine kopyalanamaz. Açıklama ekleme:\n"
            '{\n'
            '  "summary": "Mekân, faaliyet, doğrulanmış sonuçlar ve belirsizliklerin Türkçe özeti",\n'
            '  "scene_context": {"environment": {}, "activities": [], "zones": [], "context_uncertainties": []},\n'
            '  "results": [\n'
            '    {"result_id": "result-1", "result_type": "incident|contextual_finding|uncertain_observation", "time": "MM:SS", "end_time": "MM:SS", "timestamp_sec": 0.0, "duration": 0.0, "event": "açıklama", "event_type": "...", "subjects": [], "zone": "unknown", "hazard_mechanism": "zarar zinciri", "severity": "critical|high|medium|low|unknown", "evidence": {"geometric": {"supports": false, "observations": [], "limitations": []}, "visual": {"supports": false, "observations": [], "limitations": []}, "rag": {"supports": false, "observations": [], "limitations": []}, "agreement": "corroborated|single_source|conflicting|insufficient|unknown", "resolution": "kanıt/çelişki açıklaması"}, "uncertain": false, "uncertainty_reason": "", "confidence": 0.0}\n'
            '  ],\n'
            '  "risk": "Düşük|Orta|Yüksek",\n'
            '  "overall_risk": "critical|high|medium|low|unknown",\n'
            '  "actions": ["yalnız doğrulanmış sonuca uygun aksiyon"],\n'
            '  "reasoning": "bağlam, mekanizma ve kanıt harmanlama gerekçesi",\n'
            '  "uncertain": false, "uncertainty_reason": "",\n'
            '  "triggered_mock_tools": []\n'
            '}'
        )
        return "\n".join(parts)

    def _tool_catalog_text(self) -> str:
        if self.tools is None or not getattr(self.tools, "tools", None):
            return "(araç kataloğu yüklenemedi)"
        lines = []
        for name, tool in self.tools.tools.items():
            state = "" if self.tools.is_enabled(name) else " [DEVRE DIŞI]"
            lines.append(f"- {name}{state}: {tool.get('description', '')}")
        return "\n".join(lines)

    @staticmethod
    def frames_to_grid(frames: List[NDArray[np.uint8]], cols: int = 4) -> NDArray[np.uint8]:
        """Kareleri tek grid görüntüsüne birleştirir."""
        rows = (len(frames) + cols - 1) // cols
        h, w = frames[0].shape[:2]
        grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
        for idx, f in enumerate(frames):
            r, c = divmod(idx, cols)
            grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
        return grid
