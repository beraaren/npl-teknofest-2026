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
from ..models.vlm_backend import VLMBackend, create_backend
from ..utils.logger import get_logger
from .memory import ShortTermMemory
from .mock_tools import MockToolRegistry
from .rag_layer import RAGLayer


# Kanal B prompt'u: VLM'den yapılandırılmış JSON ister (S8 sözleşmesi).
# Spesifik sınıf adları (forklift, palet...) bilinçli istenmez — genel terimlerle
# doğruluk artar; spesifik tanımları algı katmanı (YOLO) sağlar.
STRUCTURED_OBSERVATION_PROMPT = (
    "Bu görüntüler bir çalışma sahasına ait video kareleridir. "
    "Gördüklerini GENEL terimlerle betimle: 'kişi/insan', 'araç', 'yük/nesne', "
    "'raf/yapı', 'sıvı/madde', 'duman veya alev' gibi; kesin tip tahmini YAPMA.\n"
    "Yanıtını SADECE aşağıdaki JSON şemasına uygun ver, açıklama ekleme:\n"
    "{\n"
    '  "scene_summary_tr": "sahne özeti",\n'
    '  "detected_entities": [{"label": "genel etiket", "confidence_hint": "low|mid|high", "notes_tr": "not"}],\n'
    '  "detected_actions_tr": ["gözlenen eylemler"],\n'
    '  "risk_flags_tr": ["dikkat çeken riskli durumlar: devrilme, düşme, duman, sızıntı, toplanma..."],\n'
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
        max_tokens: int = 512,
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
        prompt = self._build_prompt(event_signals, scene_graphs, rag_context, vlm)
        self.logger.debug(f"Karar promptu uzunluğu: {len(prompt)} karakter")

        temperature, max_tokens = self._generation_params(backend)
        raw = backend.generate(images or [], prompt, temperature=temperature, max_tokens=max_tokens)
        self.logger.debug(f"Karar ajanı çıktısı:\n{raw}")

        def retry_fn(temp: float) -> str:
            return backend.generate(images or [], prompt, temperature=temp, max_tokens=max_tokens)

        return {
            "raw_text": raw,
            "rag_risk_level": rag_context.get("risk_level", "Düşük"),
            "retry_fn": retry_fn,
        }

    def _generation_params(self, backend: VLMBackend) -> tuple[float, int]:
        name = backend.name()
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
    ) -> str:
        parts = [self.config.system_prompt]

        parts.append("\n--- ALGI KATMANI OLAY SİNYALLERİ (YOLO + kural motoru) ---")
        if event_signals:
            parts.append(json.dumps(event_signals, ensure_ascii=False, indent=2))
        else:
            parts.append("Geometrik olay sinyali tespit edilmedi.")

        parts.append("\n--- RAG KONTEXTİ (risk kataloğu eşleşmeleri) ---")
        parts.append(json.dumps(rag_context, ensure_ascii=False, indent=2))

        parts.append("\n--- VLM KARE YORUMU (bağımsız kanal) ---")
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
            "\nKanıtları düşünce akışıyla değerlendir, çelişkileri çöz ve SADECE "
            "aşağıdaki JSON şemasına uygun Türkçe yanıt ver. Açıklama ekleme:\n"
            '{\n'
            '  "summary": "Videonun genel özeti",\n'
            '  "events": [\n'
            '    {"time": "MM:SS", "event": "olay açıklaması", "event_type": "forklift_tip_over|person_fall|gathering|immobile_person|ppe_missing|dangerous_proximity|fire_smoke|leakage|...", "confidence": 0.0-1.0}\n'
            '  ],\n'
            '  "risk": "Düşük|Orta|Yüksek",\n'
            '  "actions": ["aksiyon 1", "aksiyon 2"],\n'
            '  "reasoning": "Karara nasıl ulaştığın: hangi kanıta neden güvendiğin, çelişkileri nasıl çözdüğün",\n'
            '  "confidence": 0.0-1.0,\n'
            '  "triggered_mock_tools": [\n'
            '    {"tool_name": "katalogdaki_araç_adı", "params": {"location": "...", "urgency": "Yüksek"}}\n'
            '  ]\n'
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
