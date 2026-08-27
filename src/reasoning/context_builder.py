"""Karar ajanına giden bağlam ve aday gözlem paketini deterministik üretir.

Bu modül risk seviyesi hesaplamaz. YOLO/kural motoru ve VLM çıktısını,
karar ajanının bağlam içinde değerlendirebileceği kanıt paketine dönüştürür.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List


def _seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    text = str(value or "00:00")
    try:
        minutes, seconds = text.rsplit(":", 1)
        return max(0.0, float(minutes) * 60 + float(seconds))
    except (TypeError, ValueError):
        return 0.0


def _mmss(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 60:02d}:{total % 60:02d}"


def _node_classes(scene_graphs: Iterable[Dict[str, Any]]) -> List[str]:
    classes: List[str] = []
    for graph in scene_graphs:
        nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
        if isinstance(nodes, dict):
            nodes = nodes.values()
        for node in nodes:
            if not isinstance(node, dict):
                continue
            class_name = node.get("class") or node.get("class_name")
            if class_name:
                classes.append(str(class_name))
    return classes


def build_scene_context(
    scene_graphs: List[Dict[str, Any]],
    vlm_interpretation: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Gözlenebilir mekân/faaliyet bağlamını üretir; bilinmeyeni uydurmaz."""
    vlm = vlm_interpretation or {}
    classes = Counter(_node_classes(scene_graphs))
    actions = [str(item) for item in vlm.get("detected_actions_tr", []) if str(item).strip()]
    entities = [
        str(item.get("label", ""))
        for item in vlm.get("detected_entities", [])
        if isinstance(item, dict) and str(item.get("label", "")).strip()
    ]
    scene_summary = str(vlm.get("scene_summary_tr", "")).strip()
    activities = [
        {
            "activity": action,
            "time_range": {"start_sec": 0.0, "end_sec": 0.0},
            "participants": [],
            "operational_state": "unknown",
            "evidence": ["visual"],
        }
        for action in dict.fromkeys(actions)
    ]
    uncertainties = []
    if not scene_summary:
        uncertainties.append("Bağımsız görsel sahne özeti bulunmuyor.")
    if not activities:
        uncertainties.append("Yapılan faaliyet görsel kanıttan belirlenemedi.")
    if not scene_graphs:
        uncertainties.append("Geometrik sahne grafiği bulunmuyor.")

    return {
        "environment": {
            "type": "inferred" if scene_summary or classes else "unknown",
            "description": scene_summary,
            "visible_entities": dict(classes),
            "visual_entities": list(dict.fromkeys(entities)),
            "visibility_limits": [],
        },
        "activities": activities,
        # Görüntüden güvenilir bir zone ayrımı yoksa tek bir belirsiz alan kullanılır.
        "zones": [{"id": "unknown", "purpose": "unknown", "active_operations": [], "access_constraints": "unknown"}],
        "context_uncertainties": uncertainties,
    }


def _signal_key(signal: Dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    event_type = str(signal.get("event_type") or "unknown")
    track_ids = signal.get("involved_track_ids") or []
    return event_type, tuple(sorted(str(track_id) for track_id in track_ids))


def build_candidate_observations(
    event_signals: List[Dict[str, Any]],
    vlm_interpretation: Dict[str, Any] | None,
    cluster_seconds: float = 10.0,
) -> List[Dict[str, Any]]:
    """Aynı tür/özne için kısa aralıktaki sinyalleri tek adayda kümeler.

    Kümelenme tekrarları saklar, fakat onları bağımsız olay veya bağımsız kanal
    olarak saymaz. VLM zaman yakınlığı yalnız ``nearby_visual_observations``
    olarak taşınır; severity ya da olay kimliği ödünç alınmaz.
    """
    grouped: Dict[tuple[str, tuple[str, ...]], List[Dict[str, Any]]] = {}
    for raw in event_signals:
        if not isinstance(raw, dict):
            continue
        grouped.setdefault(_signal_key(raw), []).append(raw)

    visual_events = [
        item for item in (vlm_interpretation or {}).get("risk_events", [])
        if isinstance(item, dict)
    ]
    candidates: List[Dict[str, Any]] = []
    counter = 0
    for (event_type, track_ids), signals in grouped.items():
        signals.sort(key=lambda item: _seconds(item.get("timestamp")))
        clusters: List[List[Dict[str, Any]]] = []
        for signal in signals:
            at = _seconds(signal.get("timestamp"))
            if clusters and at - _seconds(clusters[-1][-1].get("timestamp")) <= cluster_seconds:
                clusters[-1].append(signal)
            else:
                clusters.append([signal])

        for cluster in clusters:
            counter += 1
            start = _seconds(cluster[0].get("timestamp"))
            end = _seconds(cluster[-1].get("timestamp"))
            nearby_visual = []
            for visual in visual_events:
                visual_at = _seconds(visual.get("timestamp_sec"))
                if start - cluster_seconds <= visual_at <= end + cluster_seconds:
                    nearby_visual.append({
                        "timestamp_sec": visual_at,
                        "description": str(visual.get("description_tr", "")),
                        "severity": str(visual.get("severity", "unknown")),
                    })
            descriptions = list(dict.fromkeys(
                str(item.get("description", "")) for item in cluster if item.get("description")
            ))
            candidates.append({
                "candidate_id": f"candidate-{counter}",
                "event_type": event_type,
                "time_range": {"start_sec": round(start, 3), "end_sec": round(end, 3)},
                "time": _mmss(start),
                "subjects": list(track_ids),
                "zone": "unknown",
                "observation_summary": descriptions[0] if descriptions else event_type,
                "geometric_evidence": {
                    "occurrences": len(cluster),
                    "track_continuity": "unknown",
                    "confidence_values": [float(item.get("confidence", 0.0) or 0.0) for item in cluster],
                    "observations": descriptions,
                },
                "nearby_visual_observations": nearby_visual,
                "temporal_pattern": "isolated" if len(cluster) == 1 else "persistent",
            })
    return candidates
