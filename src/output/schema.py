"""Yapılandırılmış çıktı şeması (Pydantic)."""
from __future__ import annotations

from typing import Any, List, Literal

from pydantic import BaseModel, Field, field_validator


class EventEntry(BaseModel):
    time: str = Field(pattern=r"^\d{2}:\d{2}$")
    event: str
    event_type: str = "unknown"
    confidence: float = Field(ge=0.0, le=1.0)
    # VLM'den gelen zaman bilgilerini koru; UI ve atamalar için mutlak saniye gerekli.
    timestamp_sec: float = Field(ge=0.0, default=0.0)
    duration: float = Field(ge=0.0, default=0.0)
    end_time: str = Field(default="", pattern=r"^$|^\d{2}:\d{2}$")


class MockToolCall(BaseModel):
    tool_name: str
    params: dict[str, Any] = Field(default_factory=dict)


class AnalysisOutput(BaseModel):
    summary: str
    events: List[EventEntry]
    risk: Literal["Düşük", "Orta", "Yüksek"]
    actions: List[str]
    reasoning: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    triggered_mock_tools: List[MockToolCall] = Field(default_factory=list)

    @field_validator("actions")
    @classmethod
    def actions_not_empty(cls, v: List[str]) -> List[str]:
        return [a for a in v if a.strip()]

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
