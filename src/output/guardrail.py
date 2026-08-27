"""Çıktı doğrulama, retry ve semantic check."""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List

from ..config import GuardrailConfig
from ..utils.logger import get_logger
from .schema import AnalysisOutput


class OutputGuardrail:
    """VLM çıktısını şemaya zorlar, çelişkileri kontrol eder, retry yapar."""

    def __init__(self, config: GuardrailConfig):
        self.config = config
        self.logger = get_logger("OutputGuardrail")

    def validate(
        self,
        raw_text: str,
        generate_fn: Callable[[float], str],
        rag_risk_level: str = "Düşük",
    ) -> Dict[str, Any]:
        """raw_text'i parse eder; başarısız olursa temperature düşürerek retry yapar."""
        last_error = None
        for attempt, temp in enumerate(self.config.temperatures[: self.config.max_retries]):
            text = raw_text if attempt == 0 else generate_fn(temp)
            try:
                parsed = self._extract_json(text)
                parsed = self._normalize(parsed)
                output = AnalysisOutput(**parsed)
                if self.config.enable_semantic_check:
                    self._semantic_check(output, rag_risk_level)
                return output.to_dict()
            except Exception as e:
                last_error = e
                self.logger.warning(f"Guardrail attempt {attempt + 1} failed: {e}")
                continue

        self.logger.error(f"Tüm retry'ler başarısız. Son hata: {last_error}")
        return self._null_response()

    def _extract_json(self, text: str) -> Dict[str, Any]:
        # Kod bloğu içindeki JSON'u al
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        # İlk '{' ve son '}' arasını al
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("JSON bloğu bulunamadı")

        return json.loads(text[start : end + 1])

    def _normalize(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Şemaya yakın ama formatı sapmış alanları kurtarır.

        - events[].time / end_time: "0:01-0:04" gibi aralıklarda ilk zamanı alır,
          tek haneli dakikayı sıfır doldurur ("0:01" -> "00:01").
        - events[].confidence: eksikse 0.5 varsayılır.
        - events[].timestamp_sec / duration: sayıya çevrilir, eksikse 0.0.
        """
        events = parsed.get("events")
        if isinstance(events, list):
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                if "time" in ev:
                    ev["time"] = self._normalize_time(ev["time"])
                if "end_time" in ev:
                    ev["end_time"] = self._normalize_time(ev["end_time"])
                if "confidence" not in ev:
                    ev["confidence"] = 0.5
                for key in ("timestamp_sec", "duration"):
                    if key in ev:
                        try:
                            ev[key] = float(ev[key])
                        except (TypeError, ValueError):
                            ev[key] = 0.0
                    else:
                        ev[key] = 0.0
        return parsed

    @staticmethod
    def _normalize_time(value: Any) -> str:
        """Her türlü zaman ifadesini MM:SS formatına indirger."""
        match = re.search(r"(\d{1,3}):(\d{1,2})(?::(\d{1,2}))?", str(value))
        if not match:
            return str(value)
        if match.group(3) is not None:  # HH:MM:SS -> toplam MM:SS
            minutes = int(match.group(1)) * 60 + int(match.group(2))
            seconds = int(match.group(3))
        else:
            minutes, seconds = int(match.group(1)), int(match.group(2))
        return f"{minutes:02d}:{seconds:02d}"

    def _semantic_check(self, output: AnalysisOutput, rag_risk_level: str) -> None:
        """Risk seviyesi ile aksiyon/olay tutarlılığını kontrol eder."""
        risk = output.risk
        events = output.events

        # Risk yüksekse en az bir olay olmalı
        if risk == "Yüksek" and not events:
            raise ValueError("Yüksek risk ancak hiç olay yok")

        # RAG riski ile VLM riski çelişiyorsa uyarı (hard fail yerine log)
        risk_order = {"Düşük": 1, "Orta": 2, "Yüksek": 3}
        if risk_order.get(rag_risk_level, 1) > risk_order.get(risk, 1):
            self.logger.warning(
                f"VLM riski ({risk}) RAG riskinden ({rag_risk_level}) düşük; çapraz kontrol önerilir."
            )

        # Aksiyon sayısı kontrolü
        if risk == "Yüksek" and len(output.actions) < 2:
            raise ValueError("Yüksek risk için yetersiz aksiyon")

    def _null_response(self) -> Dict[str, Any]:
        return {
            "summary": self.config.null_response,
            "events": [],
            "risk": "Düşük",
            "actions": ["İnsan gözetiminde tekrar analiz yap."],
            "reasoning": "Model çıktısı şemaya uymadı.",
            "confidence": 0.0,
            "triggered_mock_tools": [],
        }
