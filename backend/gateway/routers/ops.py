"""Operasyonel router — saha uyarıları, RAG önerileri ve LLM chat."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import httpx
import yaml
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.config import load_config
from src.reasoning.rag_layer import RAGLayer

from .. import store

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ops"])

# RAGLayer sadece yaml + (opsiyonel) sentence-transformers bağımlılığı taşır;
# yokluğunda TF-IDF fallback devreye girer.
rag_layer = RAGLayer()


class RiskSegment(BaseModel):
    start_sec: int
    end_sec: int
    event_type: str


class FieldAlertCreate(BaseModel):
    camera_id: str
    risk: Literal["Düşük", "Orta", "Yüksek"]
    headline: str
    summary: str
    actions: List[str] = Field(default_factory=list)
    risk_segment: RiskSegment
    target_roles: List[str] = Field(default_factory=list)


class SuggestionQuery(BaseModel):
    event_types: List[str] = Field(default_factory=list)
    query_text: str = ""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    suggestion_id: str
    messages: List[ChatMessage]


def _get_broadcast_fn(request: Request):
    return request.app.state.broadcast_fn


@router.post("/field-alerts")
async def create_field_alert_endpoint(body: FieldAlertCreate, request: Request):
    """Yeni saha uyarısı oluşturur, DB'ye kaydeder ve WebSocket'e yayınlar."""
    risk_segment_dict = body.risk_segment.model_dump()
    row = store.create_field_alert(
        camera_id=body.camera_id,
        risk=body.risk,
        headline=body.headline,
        summary=body.summary,
        actions=body.actions,
        risk_segment=risk_segment_dict,
        target_roles=body.target_roles,
    )

    payload = {
        "id": row["id"],
        "camera_id": row["camera_id"],
        "risk": row["risk"],
        "headline": row["headline"],
        "summary": row["summary"],
        "actions": row["actions"],
        "risk_segment": row["risk_segment"],
        "target_roles": row["target_roles"],
        "created_at": row["created_at"],
    }
    broadcast_fn = _get_broadcast_fn(request)
    await broadcast_fn({"stream": "field.alert", "data": payload})
    return row


@router.get("/field-alerts")
async def list_field_alerts(
    role: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
):
    """Saha uyarılarını listeler; role ile filtreler."""
    return store.get_field_alerts(role=role, limit=limit)


@router.post("/suggestions/query")
async def query_suggestions(body: SuggestionQuery):
    """RAGLayer öneri eşleştirmesi yapar."""
    return rag_layer.match_suggestions(
        event_types=body.event_types,
        query_text=body.query_text,
        top_k=8,
    )


@router.post("/chat")
async def chat_with_suggestion(body: ChatRequest):
    """Bir İSG önerisi bağlamında LLM chat tamamlaması yapar."""
    suggestion = rag_layer.get_suggestion(body.suggestion_id)
    if suggestion is None:
        raise HTTPException(status_code=404, detail="Öneri bulunamadı")

    system_prompt = _build_chat_system_prompt(suggestion)

    config = load_config()
    base_url = config.vlm.server.base_url
    model_name = config.vlm.server.model_name

    messages = [
        {"role": "system", "content": system_prompt},
        *[{"role": m.role, "content": m.content} for m in body.messages],
    ]

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 1200,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError as exc:
        logger.error(f"VLM server unreachable: {exc}")
        raise HTTPException(
            status_code=503,
            detail=f"LLM sunucusuna bağlanılamıyor: {exc}",
        )
    except httpx.HTTPStatusError as exc:
        logger.error(f"VLM server returned error: {exc.response.status_code}")
        raise HTTPException(
            status_code=502,
            detail=f"LLM sunucusu hatası: {exc.response.status_code}",
        )
    except Exception as exc:
        logger.error(f"Chat completion failed: {exc}")
        raise HTTPException(status_code=503, detail=f"LLM isteği başarısız: {exc}")

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        logger.error(f"Unexpected LLM response shape: {exc}")
        raise HTTPException(status_code=502, detail="LLM yanıtı beklenen formatta değil")

    return {"assistant": content}


def _build_chat_system_prompt(suggestion: Dict[str, Any]) -> str:
    """Öneri + ilgili risk pattern'leri + aksiyon kataloğu ile konsolide sistem promptu."""
    baslik = suggestion.get("baslik", "")
    kategori = suggestion.get("kategori", "")
    aciklama = suggestion.get("aciklama", "")
    maliyet = suggestion.get("maliyet_tahmini", {})
    mevzuat = suggestion.get("mevzuat_referanslari", [])
    related_patterns = suggestion.get("related_patterns", [])

    lines: List[str] = [
        "Sen bir İş Sağlığı ve Güvenliği (İSG) danışmanısın. Aşağıdaki öneri,",
        "ilişkili risk pattern'leri ve aksiyon kataloğu bağlamını kullanarak",
        "kullanıcının sorusunu kısa, net ve Türkçe olarak yanıtla.",
        "",
        "ÖNERİ:",
        f"- Başlık: {baslik}",
        f"- Kategori: {kategori}",
        f"- Açıklama: {aciklama}",
    ]

    if maliyet:
        alt = maliyet.get("alt_sinir_tl", "")
        ust = maliyet.get("ust_sinir_tl", "")
        para = maliyet.get("para_birimi", "TRY")
        if alt != "" and ust != "":
            lines.append(f"- Maliyet Tahmini: {alt} - {ust} {para}")

    if mevzuat:
        lines.append("- Mevzuat Referansları:")
        for ref in mevzuat:
            lines.append(f"  • {ref}")

    patterns = rag_layer.patterns.get("patterns", {})
    if related_patterns:
        lines.extend(["", "İLGİLİ RİSK PATTERNLERİ:"])
        for name in related_patterns:
            p = patterns.get(name, {})
            if not p:
                continue
            lines.append(f"- {name}: {p.get('description', '')}")
            hazards = p.get("potential_hazards", [])
            if hazards:
                lines.append(f"  Tehlikeler: {'; '.join(hazards)}")

    actions = rag_layer.actions.get("actions", {})
    if actions and related_patterns:
        lines.extend(["", "AKSİYON KATALOĞU (Özet):"])
        for level in ("Yüksek", "Orta", "Düşük"):
            level_actions = actions.get(level, {})
            for name in related_patterns:
                specific = level_actions.get(name)
                if isinstance(specific, list) and specific:
                    lines.append(f"- [{level} | {name}] {specific[0]}")

    lines.extend([
        "",
        "Talimat: Yanıtında yukarıdaki bağlamı kullan; mevzuat referanslarına",
        "atıfta bulun; maliyet ve uygulama süresi sorulursa tahmini aralıkları belirt.",
    ])

    return "\n".join(lines)
