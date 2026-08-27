"""Karar çıktısı doğrulama, geriye uyum ve kanıt tutarlılığı."""
from __future__ import annotations

import inspect
import json
import re
from typing import Any, Callable, Dict

from ..config import GuardrailConfig
from ..utils.logger import get_logger
from .schema import AnalysisOutput


_SEVERITY_ORDER = {"unknown": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
_LEGACY_RISK = {"critical": "Yüksek", "high": "Yüksek", "medium": "Orta", "low": "Düşük", "unknown": "Düşük"}


class OutputGuardrail:
    """VLM çıktısını şemaya zorlar; kanıtsız kesinliği ve risk mirasını engeller."""

    def __init__(self, config: GuardrailConfig):
        self.config = config
        self.logger = get_logger("OutputGuardrail")

    def validate(
        self,
        raw_text: str,
        generate_fn: Callable[..., str],
        rag_risk_level: str | None = None,
    ) -> Dict[str, Any]:
        """Yanıtı doğrular.

        ``rag_risk_level`` yalnız çağıran uyumluluğu için tutulur ve kasıtlı
        olarak kullanılmaz: katalog seviyesi nihai karar veya retry sebebi değildir.
        """
        last_error = None
        for attempt, temperature in enumerate(self.config.temperatures[: self.config.max_retries]):
            text = (
                raw_text
                if attempt == 0
                else self._retry_text(generate_fn, temperature, str(last_error))
            )
            try:
                parsed = self._normalize(self._extract_json(text))
                output = AnalysisOutput(**parsed)
                if self.config.enable_semantic_check:
                    self._semantic_check(output)
                return output.to_dict()
            except Exception as exc:
                last_error = exc
                self.logger.warning("Guardrail attempt %s failed: %s", attempt + 1, exc)

        self.logger.error("Tüm retry'ler başarısız. Son hata: %s", last_error)
        return self._null_response()

    @staticmethod
    def _retry_text(
        generate_fn: Callable[..., str], temperature: float, validation_error: str
    ) -> str:
        """Yeni retry sözleşmesini kullanır, tek-parametreli eski callback'leri korur."""
        try:
            parameters = tuple(inspect.signature(generate_fn).parameters.values())
            accepts_error = any(
                parameter.kind is inspect.Parameter.VAR_POSITIONAL
                for parameter in parameters
            ) or len([
                parameter for parameter in parameters
                if parameter.kind in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                }
            ]) >= 2
        except (TypeError, ValueError):
            accepts_error = False
        if accepts_error:
            return generate_fn(temperature, validation_error)
        return generate_fn(temperature)

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("JSON bloğu bulunamadı")
        return json.loads(text[start : end + 1])

    def _normalize(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Eski ``events`` ve yeni ``results`` sözleşmesini eşzamanlı destekler."""
        results = parsed.get("results")
        events = parsed.get("events")
        if not isinstance(results, list):
            results = []
        if not isinstance(events, list):
            events = []

        # Eski model yanıtı: her legacy event bir incident adayıdır; kanıt/severity
        # uydurulmaz, bu yüzden varsayılanlar unknown kalır.
        if not results:
            for index, event in enumerate(events, start=1):
                if not isinstance(event, dict):
                    continue
                results.append({
                    "result_id": f"legacy-{index}",
                    "result_type": "incident",
                    "time": event.get("time", "00:00"),
                    "end_time": event.get("end_time", ""),
                    "timestamp_sec": event.get("timestamp_sec", 0.0),
                    "duration": event.get("duration", 0.0),
                    "event": event.get("event", ""),
                    "event_type": event.get("event_type", "unknown"),
                    "severity": event.get("severity", "unknown"),
                    "confidence": event.get("confidence", 0.5),
                    "evidence": {"agreement": "unknown"},
                })

        normalized_results = []
        for index, result in enumerate(results, start=1):
            if not isinstance(result, dict):
                continue
            item = dict(result)
            item["result_id"] = str(item.get("result_id") or f"result-{index}")
            item["time"] = self._normalize_time(item.get("time", "00:00"))
            if not re.fullmatch(r"\d{2}:\d{2}", item["time"]):
                item["time"] = "00:00"
            if item.get("end_time"):
                item["end_time"] = self._normalize_time(item["end_time"])
            else:
                item["end_time"] = ""
            for key in ("timestamp_sec", "duration", "confidence"):
                try:
                    item[key] = float(item.get(key, 0.0 if key != "confidence" else 0.5))
                except (TypeError, ValueError):
                    item[key] = 0.0 if key != "confidence" else 0.5
            item["timestamp_sec"] = max(0.0, item["timestamp_sec"])
            item["duration"] = max(0.0, item["duration"])
            item["confidence"] = min(1.0, max(0.0, item["confidence"]))
            item.setdefault("result_type", "incident")
            item.setdefault("severity", "unknown")
            item.setdefault("subjects", [])
            item.setdefault("zone", "unknown")
            item.setdefault("hazard_mechanism", "")
            item.setdefault("uncertain", False)
            item.setdefault("uncertainty_reason", "")
            item.setdefault("evidence", {"agreement": "unknown"})
            normalized_results.append(item)
        parsed["results"] = normalized_results

        # Legacy consumers yalnız incidentleri görür. Contextual/uncertain sonuç
        # replay ya da saha atamasına düşmez.
        if not events:
            events = [
                {
                    "time": result["time"],
                    "end_time": result["end_time"],
                    "timestamp_sec": result["timestamp_sec"],
                    "duration": result["duration"],
                    "event": result.get("event", ""),
                    "event_type": result.get("event_type", "unknown"),
                    "confidence": result["confidence"],
                    "result_id": result["result_id"],
                    "severity": result["severity"],
                }
                for result in normalized_results
                if result.get("result_type") == "incident"
            ]
        for event in events:
            if not isinstance(event, dict):
                continue
            event["time"] = self._normalize_time(event.get("time", "00:00"))
            event["end_time"] = self._normalize_time(event["end_time"]) if event.get("end_time") else ""
            event.setdefault("confidence", 0.5)
            event.setdefault("timestamp_sec", 0.0)
            event.setdefault("duration", 0.0)
            event.setdefault("severity", "unknown")
            event.setdefault("result_id", "")
            for key in ("timestamp_sec", "duration", "confidence"):
                try:
                    event[key] = float(event[key])
                except (TypeError, ValueError):
                    event[key] = 0.5 if key == "confidence" else 0.0
        parsed["events"] = events

        max_severity = max((item.get("severity", "unknown") for item in normalized_results), key=lambda value: _SEVERITY_ORDER.get(value, 0), default="unknown")
        parsed.setdefault("overall_risk", max_severity)
        parsed.setdefault("risk", _LEGACY_RISK.get(parsed["overall_risk"], "Düşük"))
        parsed.setdefault("scene_context", {})
        parsed.setdefault("uncertain", bool(normalized_results) and all(item.get("uncertain") for item in normalized_results))
        parsed.setdefault("uncertainty_reason", "")
        parsed.setdefault("actions", [])
        parsed.setdefault("reasoning", "")
        parsed.setdefault("confidence", 0.0)
        parsed.setdefault("triggered_mock_tools", [])

        # Bazı VLM yanıtları araç çağrılarını yalnız araç adı olarak döndürüyor.
        # Şemayı ve aşağı akıştaki araç yürütücüsünü korumak için bu eski biçimi
        # Pydantic doğrulamasından önce kanonik çağrı nesnesine dönüştür.
        tool_calls = parsed["triggered_mock_tools"]
        if isinstance(tool_calls, str):
            tool_calls = [tool_calls]
        if isinstance(tool_calls, list):
            parsed["triggered_mock_tools"] = [
                {"tool_name": tool_call, "params": {}}
                if isinstance(tool_call, str)
                else tool_call
                for tool_call in tool_calls
            ]
        return parsed

    @staticmethod
    def _normalize_time(value: Any) -> str:
        match = re.search(r"(\d{1,3}):(\d{1,2})(?::(\d{1,2}))?", str(value))
        if not match:
            return str(value)
        if match.group(3) is not None:
            minutes, seconds = int(match.group(1)) * 60 + int(match.group(2)), int(match.group(3))
        else:
            minutes, seconds = int(match.group(1)), int(match.group(2))
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _semantic_check(output: AnalysisOutput) -> None:
        for result in output.results:
            if result.result_type == "uncertain_observation" and not result.uncertainty_reason:
                raise ValueError("Belirsiz gözlem için uncertainty_reason zorunlu")
            if result.severity in {"high", "critical"}:
                if not result.hazard_mechanism:
                    raise ValueError("Yüksek/kritik sonuçta hazard_mechanism zorunlu")
                if result.evidence.agreement in {"insufficient", "unknown"}:
                    raise ValueError("Yüksek/kritik sonuçta yeterli kanıt anlaşması zorunlu")
            if result.result_type == "contextual_finding" and result.duration < 0:
                raise ValueError("Bağlamsal bulgu süresi negatif olamaz")

    def _null_response(self) -> Dict[str, Any]:
        return {
            "summary": self.config.null_response,
            "scene_context": {},
            "results": [{
                "result_id": "null-response",
                "result_type": "uncertain_observation",
                "time": "00:00",
                "event": "Karar üretilemedi.",
                "event_type": "unknown",
                "severity": "unknown",
                "uncertain": True,
                "uncertainty_reason": "Model çıktısı şemaya uymadı.",
                "evidence": {"agreement": "insufficient"},
            }],
            "events": [],
            "risk": "Düşük",
            "overall_risk": "unknown",
            "actions": ["İnsan gözetiminde tekrar analiz yap."],
            "reasoning": "Model çıktısı şemaya uymadı.",
            "uncertain": True,
            "uncertainty_reason": "Model çıktısı şemaya uymadı.",
            "confidence": 0.0,
            "triggered_mock_tools": [],
        }
