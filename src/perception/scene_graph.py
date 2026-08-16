"""Scene graph: nesneler (düğüm) ve ilişkiler (kenar)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

from .detector import Detection


@dataclass
class SceneNode:
    node_id: str
    class_name: str
    track_id: int | None
    bbox: tuple[float, float, float, float]
    confidence: float
    frame_idx: int

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class SceneEdge:
    source: str
    target: str
    relation: str
    weight: float


class SceneGraph:
    """Bir karedeki nesneler ve aralarındaki ilişkiler."""

    def __init__(self, frame_idx: int = 0, timestamp: float = 0.0):
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.nodes: Dict[str, SceneNode] = {}
        self.edges: List[SceneEdge] = []

    def add_node(self, node: SceneNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: SceneEdge) -> None:
        self.edges.append(edge)

    def build_relations(self, proximity_threshold: float = 100.0) -> None:
        """Yakınlık ve taşıma ilişkilerini çıkarır."""
        node_list = list(self.nodes.values())
        for i, a in enumerate(node_list):
            for b in node_list[i + 1 :]:
                dist = math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])
                if dist <= proximity_threshold:
                    self.add_edge(SceneEdge(a.node_id, b.node_id, "near", 1.0 - dist / proximity_threshold))

                # forklift palet taşıyor olabilir
                if {a.class_name, b.class_name} == {"arac", "palet"} and dist <= proximity_threshold * 1.5:
                    self.add_edge(SceneEdge(a.node_id, b.node_id, "carrying", 0.8))

                # insan baret/yelek giyiyor olabilir
                if a.class_name == "insan" and b.class_name in ("baret", "yelek") and dist <= proximity_threshold * 0.8:
                    self.add_edge(SceneEdge(a.node_id, b.node_id, "wearing", 0.9))

    def find_nodes(self, class_name: str) -> List[SceneNode]:
        return [n for n in self.nodes.values() if n.class_name == class_name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": round(self.timestamp, 2),
            "nodes": [
                {
                    "id": n.node_id,
                    "class": n.class_name,
                    "track_id": n.track_id,
                    "center": [round(n.center[0], 2), round(n.center[1], 2)],
                    "confidence": round(n.confidence, 3),
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {"source": e.source, "target": e.target, "relation": e.relation, "weight": round(e.weight, 3)}
                for e in self.edges
            ],
        }

    @classmethod
    def from_detections(cls, frame_idx: int, timestamp: float, detections: List[Detection]) -> SceneGraph:
        graph = cls(frame_idx=frame_idx, timestamp=timestamp)
        for idx, det in enumerate(detections):
            tid = getattr(det, "track_id", None)
            node_id = f"{det.class_name}_{tid}" if tid is not None else f"{det.class_name}_{idx}"
            graph.add_node(
                SceneNode(
                    node_id=node_id,
                    class_name=det.class_name,
                    track_id=tid,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    frame_idx=frame_idx,
                )
            )
        graph.build_relations()
        return graph
