"""Karar ajanının yapılandırılmış çıktı sözleşmesi."""
from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, field_validator


ResultType = Literal["incident", "contextual_finding", "uncertain_observation"]
Severity = Literal["critical", "high", "medium", "low", "unknown"]
EvidenceAgreement = Literal["corroborated", "single_source", "conflicting", "insufficient", "unknown"]


class EvidenceSource(BaseModel):
    """Bir karar kanalının desteği ve sınırları.

    ``supports`` yalnızca bu kanalın gözlemi aday sonucu destekliyorsa doğrudur.
    RAG referansı olayın gerçekleştiğini kanıtlamaz; yalnızca uygulanabilir bir
    tehlike mekanizmasını/prosedürü işaretler.
    """

    supports: bool = False
    observations: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class ResultEvidence(BaseModel):
    """Sonucun çok-kaynaklı kanıt izi."""

    geometric: EvidenceSource = Field(default_factory=EvidenceSource)
    visual: EvidenceSource = Field(default_factory=EvidenceSource)
    rag: EvidenceSource = Field(default_factory=EvidenceSource)
    agreement: EvidenceAgreement = "unknown"
    resolution: str = ""


class ResultEntry(BaseModel):
    """Karar ajanının kanonik sonucu.

    ``incident`` replay ve saha ataması için zamanlanabilir sonuçtur.
    ``contextual_finding`` kalıcı fakat bağlama uygun bulgudur; tek başına
    alarm penceresi açmaz. ``uncertain_observation`` insan incelemesi ister.
    """

    result_id: str = ""
    result_type: ResultType = "incident"
    time: str = Field(default="00:00", pattern=r"^\d{2}:\d{2}$")
    end_time: str = Field(default="", pattern=r"^$|^\d{2}:\d{2}$")
    timestamp_sec: float = Field(ge=0.0, default=0.0)
    duration: float = Field(ge=0.0, default=0.0)
    event: str
    event_type: str = "unknown"
    subjects: List[str] = Field(default_factory=list)
    zone: str = "unknown"
    hazard_mechanism: str = ""
    severity: Severity = "unknown"
    evidence: ResultEvidence = Field(default_factory=ResultEvidence)
    uncertain: bool = False
    uncertainty_reason: str = ""
    # Yalnız geriye uyum ve iç metrik için saklanır; UI bunu sayısal göstermez.
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class EventEntry(BaseModel):
    """Eski replay/atama sözleşmesi için incident projeksiyonu."""

    time: str = Field(pattern=r"^\d{2}:\d{2}$")
    event: str
    event_type: str = "unknown"
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp_sec: float = Field(ge=0.0, default=0.0)
    duration: float = Field(ge=0.0, default=0.0)
    end_time: str = Field(default="", pattern=r"^$|^\d{2}:\d{2}$")
    result_id: str = ""
    severity: Severity = "unknown"


class MockToolCall(BaseModel):
    tool_name: str
    params: dict[str, Any] = Field(default_factory=dict)


class AnalysisOutput(BaseModel):
    """Nihai karar: ``results`` kanoniktir, ``events`` geçiş uyumluluğudur."""

    summary: str
    scene_context: Dict[str, Any] = Field(default_factory=dict)
    results: List[ResultEntry] = Field(default_factory=list)
    events: List[EventEntry] = Field(default_factory=list)
    # Mevcut servis/UI kontratı için Türkçe üçlü genel risk korunur.
    risk: Literal["Düşük", "Orta", "Yüksek"] = "Düşük"
    overall_risk: Severity = "unknown"
    actions: List[str] = Field(default_factory=list)
    reasoning: str = ""
    uncertain: bool = False
    uncertainty_reason: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    triggered_mock_tools: List[MockToolCall] = Field(default_factory=list)

    @field_validator("actions")
    @classmethod
    def actions_not_empty(cls, value: List[str]) -> List[str]:
        return [action for action in value if action.strip()]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
