"""Mock tool kataloğu ve çağrı yöneticisi."""
from __future__ import annotations

from typing import Any, Dict, List

import yaml

from ..config import get_data_path
from ..utils.logger import get_logger


class MockToolRegistry:
    """Karar ajanının çağırabileceği mock fonksiyonları tanımlar ve çalıştırır."""

    def __init__(self, tools_path: str | None = None):
        self.logger = get_logger("MockTools")
        path = tools_path or get_data_path("mock_tools.yaml")
        self.tools: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self.tools = {t["name"]: t for t in data.get("tools", [])} if isinstance(data.get("tools"), list) else data.get("tools", {})

    def is_enabled(self, tool_name: str) -> bool:
        tool = self.tools.get(tool_name, {})
        return bool(tool.get("enabled", True))

    def execute(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self.tools:
            return {"status": "error", "message": f"Bilinmeyen araç: {tool_name}"}
        if not self.is_enabled(tool_name):
            return {"status": "disabled", "message": f"{tool_name} şu anda devre dışı."}

        self.logger.info(f"Mock tool çağrıldı: {tool_name} | params={params}")
        return {
            "status": "success",
            "tool": tool_name,
            "params": params,
            "mock_result": f"{tool_name} başarıyla simüle edildi.",
        }

    def suggest_tools(self, risk_level: str, event_types: List[str]) -> List[Dict[str, Any]]:
        """Risk seviyesi ve olay tipine göre önerilen mock tool'ları döner."""
        suggestions = []
        mapping = {
            "Yüksek": ["call_health_team", "secure_area", "record_incident", "notify_supervisor", "lockdown_facility", "trigger_fire_suppression", "activate_cbrn_protocol"],
            "Orta": ["record_incident", "notify_supervisor", "sound_alarm"],
            "Düşük": ["record_incident"],
        }
        for tool_name in mapping.get(risk_level, []):
            if self.is_enabled(tool_name):
                tool = self.tools.get(tool_name, {})
                suggestions.append({
                    "tool_name": tool_name,
                    "description": tool.get("description", ""),
                    "params": tool.get("params", {}),
                })
        return suggestions
