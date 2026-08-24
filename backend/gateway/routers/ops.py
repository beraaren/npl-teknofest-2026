"""Operasyonel router — saha uyarıları, RAG önerileri ve LLM chat."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import httpx
import yaml
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.config import load_config
from src.reasoning.mock_tools import MockToolRegistry
from src.reasoning.rag_layer import RAGLayer

from .. import library, roles, store

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ops"])

# RAGLayer sadece yaml + (opsiyonel) sentence-transformers bağımlılığı taşır;
# yokluğunda TF-IDF fallback devreye girer.
rag_layer = RAGLayer()

# Aksiyon butonlarının gerçekten çalıştırdığı araç kataloğu. Arayüzdeki
# butonlar eskiden yalnızca bir bildirim gösteriyordu; artık bu kayıt üzerinden
# gerçek (simüle) çalıştırma yapılır ve sonuç denetim izine yazılır.
tool_registry = MockToolRegistry()


def _server_config():
    """``config.vlm.server`` bloğunu döner (her istekte yeniden okunmaz)."""
    global _SERVER_CONFIG
    if _SERVER_CONFIG is None:
        _SERVER_CONFIG = load_config().vlm.server
    return _SERVER_CONFIG


_SERVER_CONFIG = None


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


class AssignmentCreate(BaseModel):
    """Süpervizörün bir olayı bir ekibe atama isteği.

    Ekranda gösterilecek alanlar (özet, risk, olay anı, aksiyonlar) sunucu
    tarafında analiz dosyasından okunur; istemcinin bunları göndermesi
    gerekmez ve gönderse de güvenilmez. Böylece saha ekibinin gördüğü metin,
    süpervizörün gördüğü metinle **aynı kaynaktan** gelir.
    """
    analysis_slug: str
    role: str
    camera_id: str = ""
    # Hangi olaya atama yapıldığı; verilmezse en kritik olay seçilir.
    event_index: Optional[int] = None
    note: str = ""
    assigned_by: str = "supervisor"


class AssignmentStatusUpdate(BaseModel):
    status: Literal["atandi", "goruldu", "devam_ediyor", "tamamlandi"]


class ToolExecuteRequest(BaseModel):
    """Arayüzdeki bir aksiyon butonunun tetiklediği araç çalıştırma isteği."""
    tool_name: str
    params: Dict[str, Any] = Field(default_factory=dict)
    camera_id: str = ""
    analysis_slug: str = ""
    job_id: str = ""
    # Denetim izi için: aksiyonu hangi ekran tetikledi.
    triggered_by: Literal["supervisor", "field", "system"] = "supervisor"


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


@router.post("/assignments", status_code=201)
async def create_assignment_endpoint(body: AssignmentCreate, request: Request):
    """Bir olayı belirli bir ekibe atar ve o ekibin ekranına düşürür.

    Gösterilecek metinler (ajan özeti, risk, olay anı, önerilen aksiyonlar)
    **sunucuda** analiz dosyasından okunur. İstemci bunları göndermez; böylece
    saha ekibinin gördüğü özet, süpervizörün gördüğüyle aynı kaynaktan gelir ve
    ekranlar arasında sapma olmaz.

    Raises:
        HTTPException: Analiz bulunamazsa 404.
    """
    analysis = library.get(body.analysis_slug)
    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail=f"Analiz bulunamadı: {body.analysis_slug}. "
                   f"Kütüphanede {library.count()} analiz var.",
        )

    # Rol adı kanonik yazıma çevrilir; aksi hâlde "sağlık" ile "saglik" ayrı
    # kayıtlar oluşturur ve saha ekibi kendi görevini göremez.
    role = roles.normalize_role(body.role)
    if not role:
        raise HTTPException(status_code=400, detail="Rol belirtilmedi.")

    event = library.pick_event(analysis, body.event_index)

    row = store.create_assignment(
        analysis_slug=body.analysis_slug,
        role=role,
        camera_id=body.camera_id,
        risk=str(analysis.get("risk") or ""),
        headline=str(analysis.get("headline") or ""),
        # Karar ajanının yazdığı olay özeti — iki ekranda da aynı metin.
        summary=str(analysis.get("summary") or ""),
        reasoning=str(analysis.get("reasoning") or ""),
        event_type=str(event.get("event_type") or ""),
        event_seconds=float(event.get("seconds") or 0.0),
        event_timestamp=str(event.get("timestamp") or ""),
        actions=list(analysis.get("actions") or []),
        video_file=str(analysis.get("video_file") or ""),
        note=body.note,
        assigned_by=body.assigned_by,
    )

    broadcast_fn = _get_broadcast_fn(request)
    await broadcast_fn({"stream": "assignment.created", "data": row})

    logger.info(
        f"Atama oluşturuldu: #{row['id']} {body.role} <- {body.analysis_slug} "
        f"({row['event_type']} @ {row['event_timestamp']})"
    )
    return row


@router.get("/assignments")
async def list_assignments(
    role: Optional[str] = Query(default=None, description="Saha ekranı kendi rolünü verir"),
    status: Optional[str] = Query(default=None),
    analysis_slug: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
):
    """Atamaları listeler.

    Saha ekranı ``role`` ile çağırır ve **yalnızca kendisine atananları** alır;
    rol verilmezse (süpervizör görünümü) tüm atamalar döner.
    """
    return store.get_assignments(
        role=roles.normalize_role(role) or None,
        status=status,
        analysis_slug=analysis_slug,
        limit=limit,
    )


@router.get("/roles")
async def list_roles():
    """Saha ekibi rollerini ve görünen etiketlerini döner.

    Arayüz rol seçicisini bu listeden kurar; böylece istemci ile sunucu aynı
    kanonik rol kimliklerini kullanır.
    """
    return roles.all_roles()


@router.get("/assignments/counts")
async def assignment_counts():
    """Rol ve durum bazında atama sayıları (süpervizör özet paneli)."""
    return store.get_assignment_counts()


@router.patch("/assignments/{assignment_id}")
async def update_assignment(
    assignment_id: int, body: AssignmentStatusUpdate, request: Request
):
    """Atamanın durumunu ilerletir (``goruldu`` / ``tamamlandi``).

    Saha ekibinin işaretlemesi veritabanına yazılır ve yayınlanır; böylece
    süpervizör ekranı da durumu görür. Eskiden bu tür etkileşimler yalnızca
    ekranda geçici bir bildirim gösteriyordu.

    Raises:
        HTTPException: Atama yoksa 404, durum geçersizse 400.
    """
    try:
        row = store.update_assignment_status(assignment_id, body.status)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if row is None:
        raise HTTPException(status_code=404, detail=f"Atama bulunamadı: {assignment_id}")

    broadcast_fn = _get_broadcast_fn(request)
    await broadcast_fn({"stream": "assignment.updated", "data": row})
    return row


@router.get("/tools")
async def list_tools():
    """Çalıştırılabilir araç kataloğunu döner.

    Arayüz, aksiyon butonlarını bu listeye göre üretir; katalogda olmayan bir
    araç adı gönderilmesi engellenir.
    """
    return [
        {
            "tool_name": name,
            "description": tool.get("description", ""),
            "params": tool.get("params", {}),
            "enabled": tool_registry.is_enabled(name),
        }
        for name, tool in (tool_registry.tools or {}).items()
    ]


@router.post("/tools/execute")
async def execute_tool(body: ToolExecuteRequest, request: Request):
    """Bir aracı gerçekten çalıştırır, denetim izine yazar ve yayınlar.

    Arayüzdeki aksiyon butonları eskiden yalnızca ekranda bir bildirim
    gösteriyordu; hiçbir şey çalışmıyordu. Bu uç nokta aracı
    :class:`~src.reasoning.mock_tools.MockToolRegistry` üzerinden çalıştırır,
    sonucu ``events`` tablosuna kaydeder ve ``tool.executed`` akışına yayınlar
    ki süpervizör ve saha ekranları aynı sonucu görsün.

    Raises:
        HTTPException: Araç katalogda yoksa 404, devre dışıysa 409.
    """
    catalog = tool_registry.tools or {}
    if body.tool_name not in catalog:
        raise HTTPException(
            status_code=404,
            detail=f"Araç katalogda yok: {body.tool_name}. "
                   f"Geçerli araçlar: {', '.join(catalog)}",
        )
    if not tool_registry.is_enabled(body.tool_name):
        raise HTTPException(status_code=409, detail=f"Araç devre dışı: {body.tool_name}")

    result = tool_registry.execute(body.tool_name, body.params)

    payload = {
        "job_id": body.job_id or body.camera_id or "manuel",
        "camera_id": body.camera_id,
        "analysis_slug": body.analysis_slug,
        "tool_name": body.tool_name,
        "params": body.params,
        "status": result.get("status", "unknown"),
        "mock_result": result.get("mock_result", result.get("message", "")),
        "triggered_by": body.triggered_by,
        "created_at": datetime.now().isoformat(),
    }

    if payload["job_id"]:
        store.save_event(payload["job_id"], "tool.executed", payload)

    broadcast_fn = _get_broadcast_fn(request)
    await broadcast_fn({"stream": "tool.executed", "data": payload})

    logger.info(
        f"Araç çalıştırıldı: {body.tool_name} "
        f"(kamera={body.camera_id or '-'}, tetikleyen={body.triggered_by})"
    )
    return payload


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

    cfg = _server_config()
    base_url = cfg.base_url.rstrip("/")
    model_name = cfg.model_name

    messages = [
        {"role": "system", "content": system_prompt},
        *[{"role": m.role, "content": m.content} for m in body.messages],
    ]

    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 1200,
    }
    # Akıl yürütme kapalı: açıkken ayrıştırıcı düşünme izini sildiği için yanıt
    # HTTP 200 ile boş dönebiliyor (bkz. src/models/vlm_backend.py notu).
    if cfg.enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": cfg.enable_thinking}

    # Anahtar yapılandırmada değil ortam değişkenindedir (.env).
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(cfg.api_key_env, "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        logger.warning(
            f"'{cfg.api_key_env}' tanımlı değil; chat isteği kimlik doğrulaması "
            f"olmadan gidecek ve servis 401 döndürebilir."
        )

    try:
        async with httpx.AsyncClient(timeout=float(cfg.timeout_sec)) as client:
            # DİKKAT: base_url zaten "/v1" ile biter. Buraya bir kez daha "/v1"
            # eklenmesi ".../v1/v1/chat/completions" üretip 404'e yol açıyordu.
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
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
