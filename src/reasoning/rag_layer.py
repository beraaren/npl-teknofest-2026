"""Çift indeksli RAG katmanı: risk pattern'leri (vektör indeksi) + aksiyon kataloğu.

Ana sorgu ObserverAgent'ın gözlem raporudur; event_signals ikincil filtre/boost
olarak kullanılır. Vektör arama saf Python TF-IDF + kosinüs benzerliğidir —
bağımlılık yok, yaklaşık 30 satır.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml

from ..config import get_data_path

_TOKEN_RE = re.compile(r"[a-zçğıöşü0-9_]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _tfidf_index(docs: Dict[str, str]) -> tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Doküman adı -> {token: tfidf} (L2-normalize) ve token -> idf döner."""
    tf: Dict[str, Dict[str, float]] = {}
    df: Dict[str, int] = {}
    for name, text in docs.items():
        counts: Dict[str, float] = {}
        for tok in _tokenize(text):
            counts[tok] = counts.get(tok, 0.0) + 1.0
        tf[name] = counts
        for tok in counts:
            df[tok] = df.get(tok, 0) + 1
    n = max(len(docs), 1)
    idf = {tok: math.log((n + 1) / (d + 1)) + 1.0 for tok, d in df.items()}

    def _vec(counts: Dict[str, float]) -> Dict[str, float]:
        v = {tok: c * idf.get(tok, 0.0) for tok, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {tok: x / norm for tok, x in v.items()}

    return {name: _vec(c) for name, c in tf.items()}, idf


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(tok, 0.0) for tok, w in a.items())


class RAGLayer:
    """Gözlem raporunu risk pattern'leriyle eşleştirir, aksiyon kataloğundan öneri üretir."""

    def __init__(
        self,
        patterns_path: str | None = None,
        actions_path: str | None = None,
    ):
        self.patterns: Dict[str, Any] = {}
        self.actions: Dict[str, Any] = {}

        ppath = Path(patterns_path) if patterns_path else get_data_path("risk_patterns.yaml")
        if ppath.exists():
            with open(ppath, "r", encoding="utf-8") as f:
                self.patterns = yaml.safe_load(f) or {}

        apath = Path(actions_path) if actions_path else get_data_path("action_catalog.yaml")
        if apath.exists():
            with open(apath, "r", encoding="utf-8") as f:
                self.actions = yaml.safe_load(f) or {}

        # Vektör indeksi: her pattern'in description + indicators metni bir doküman.
        docs = {
            name: f"{name} {p.get('description', '')} {' '.join(p.get('indicators', []))}"
            for name, p in self.patterns.get("patterns", {}).items()
        }
        self._index, self._idf = _tfidf_index(docs)

    def _query_vector(self, text: str) -> Dict[str, float]:
        counts: Dict[str, float] = {}
        for tok in _tokenize(text):
            counts[tok] = counts.get(tok, 0.0) + 1.0
        v = {tok: c * self._idf.get(tok, 0.0) for tok, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {tok: x / norm for tok, x in v.items()}

    def match_patterns(self, event_signals: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Sinyal event_type'larıyla pattern adı eşleşmesi (ikincil filtre/boost).

        Dönüş: pattern adı -> {"signal": sig}. event_type == pattern adı ya da
        biri diğerini içeriyorsa eşleşme sayılır.
        """
        matched: Dict[str, Dict[str, Any]] = {}
        for sig in event_signals:
            event_type = sig.get("event_type", "")
            for name in self.patterns.get("patterns", {}):
                if event_type == name or event_type in name or name in event_type:
                    matched[name] = {"signal": sig}
        return matched

    def build_context(
        self,
        observation_report: Any,
        event_signals: List[Dict[str, Any]],
        threshold: float = 0.1,
        boost: float = 1.5,
    ) -> Dict[str, Any]:
        """Karar ajanına verilecek RAG kontekstini oluşturur.

        Ana sorgu: observation_report (ObserverAgent raporu, metne çevrilir).
        İkincil filtre: event_signals — sinyalle eşleşen pattern'ler boost'lanır.
        """
        query = observation_report if isinstance(observation_report, str) else json.dumps(
            observation_report, ensure_ascii=False
        )
        qv = self._query_vector(query)
        boosted = self.match_patterns(event_signals)

        matches: List[Dict[str, Any]] = []
        for name, pattern in self.patterns.get("patterns", {}).items():
            sim = _cosine(qv, self._index.get(name, {}))
            signal_hit = boosted.get(name)
            if signal_hit:
                sim = max(sim * boost, threshold)  # sinyal doğrudan pattern'i işaret ediyor
            if sim < threshold:
                continue
            entry = {
                "pattern": name,
                "description": pattern.get("description", ""),
                "risk_score": pattern.get("risk_score", 0),
                "risk_level": pattern.get("risk_level", "Düşük"),
                "similarity": round(sim, 3),
            }
            if signal_hit:
                entry["matched_signal"] = signal_hit["signal"]
            matches.append(entry)

        if not matches:
            return {"risk_level": "Düşük", "risk_score": 0, "actions": [], "matched_patterns": []}

        top = max(matches, key=lambda m: m.get("risk_score", 0))
        risk_level = top.get("risk_level", "Düşük")
        actions = self.recommend_actions(risk_level, [m["pattern"] for m in matches])

        return {
            "risk_level": risk_level,
            "risk_score": top.get("risk_score", 0),
            "actions": actions,
            "matched_patterns": matches,
        }

    def recommend_actions(self, risk_level: str, event_types: List[str]) -> List[str]:
        """Risk seviyesi ve olay tiplerine göre aksiyon önerir (önce özel, sonra default)."""
        catalog = self.actions.get("actions", {})
        level_actions = catalog.get(risk_level, {})

        actions: List[str] = []
        for et in event_types:
            specific = level_actions.get(et)
            if isinstance(specific, list):
                actions.extend(specific)

        default = level_actions.get("default", [])
        if isinstance(default, list):
            actions.extend(default)

        seen = set()
        unique = []
        for a in actions:
            if a not in seen:
                seen.add(a)
                unique.append(a)
        return unique
