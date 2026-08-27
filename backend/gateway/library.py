"""Analiz kütüphanesi erişim katmanı — kaydedilmiş sonuçları okur.

Arayüz canlı bir kamera sistemi gibi davranır ama çalışma anında **hiçbir model
çağrısı yapmaz**. Analizler ``scripts/analyze_video_library.py`` ile önceden
üretilir; bu modül o çıktıları okuyup replay motoru, atama uçları ve video
servisi için tek bir erişim noktası sunar.

Neden tek modül: üç ayrı tüketici (replay motoru, atama uçları, video servisi)
aynı dosyaları okuyor. Yükleme ve dosya adı eşlemesi burada toplanmazsa aynı
mantık üç yerde tekrarlanır ve tutarsızlık kaçınılmaz olur.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ANALYSES_DIR = REPO_ROOT / "data" / "library" / "analyses"
VIDEOS_DIR = REPO_ROOT / "videos"

# Analizler süreç ömrü boyunca bellekte tutulur; her istekte diskten okumak
# gereksizdir çünkü dosyalar çalışma sırasında değişmez (offline üretilir).
_cache: Dict[str, dict] = {}
_lock = threading.Lock()
_loaded = False


def _resolve_video(video_rel: str) -> Optional[Path]:
    """Analizdeki göreli video yolunu diskte bulur.

    Yol doğrudan bulunamazsa büyük/küçük harf duyarsız arama yapılır. Buna
    ihtiyaç var çünkü analizler Windows'ta üretilebiliyor (dosya sistemi harf
    duyarsız) ama servis Linux kapsayıcısında çalışıyor: diskteki klasör
    ``Videos`` iken analizde ``videos/...`` yazılıysa yol Linux'ta çözülemez ve
    tüm kütüphane sessizce boş görünür.

    Args:
        video_rel: Analizdeki depo-göreli video yolu.

    Returns:
        Bulunan dosyanın tam yolu veya ``None``.
    """
    if not video_rel:
        return None

    direct = REPO_ROOT / video_rel
    if direct.exists():
        return direct

    parts = Path(video_rel).parts
    if not parts:
        return None

    # Her seviyeyi harf duyarsız eşleştirerek ilerle.
    current = REPO_ROOT
    for part in parts:
        if not current.is_dir():
            return None
        match = next(
            (child for child in current.iterdir() if child.name.lower() == part.lower()),
            None,
        )
        if match is None:
            return None
        current = match
    return current if current.exists() else None


def _load_locked() -> None:
    """Kilit altında kütüphaneyi diskten yükler."""
    global _loaded
    _cache.clear()

    if not ANALYSES_DIR.exists():
        logger.warning(
            f"Analiz klasörü yok: {ANALYSES_DIR}. "
            f"Önce 'python scripts/analyze_video_library.py' çalıştırılmalı."
        )
        _loaded = True
        return

    for path in sorted(ANALYSES_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Analiz okunamadı, atlanıyor: {path.name} ({exc})")
            continue

        slug = data.get("slug") or path.stem
        video_rel = data.get("video_file") or ""
        video_abs = _resolve_video(video_rel)

        # Videosu olmayan analiz arayüzde kırık kart üretir; baştan dışlanır.
        if video_abs is None:
            logger.warning(
                f"'{slug}' analizinin videosu bulunamadı ({video_rel}); atlanıyor."
            )
            continue

        data["_video_abs"] = str(video_abs)
        _cache[slug] = data

    _loaded = True
    logger.info(f"Analiz kütüphanesi yüklendi: {len(_cache)} kayıt ({ANALYSES_DIR})")


def ensure_loaded() -> None:
    """Kütüphane henüz yüklenmediyse yükler."""
    if _loaded:
        return
    with _lock:
        if not _loaded:
            _load_locked()


def reload() -> int:
    """Kütüphaneyi diskten yeniden okur ve kayıt sayısını döner."""
    with _lock:
        _load_locked()
    return len(_cache)


def all_analyses() -> List[dict]:
    """Tüm analizleri slug sırasıyla döner."""
    ensure_loaded()
    return [_cache[k] for k in sorted(_cache)]


def slugs() -> List[str]:
    """Kullanılabilir analiz kimliklerini döner."""
    ensure_loaded()
    return sorted(_cache)


def get(slug: str) -> Optional[dict]:
    """Tek bir analizi döner; yoksa ``None``."""
    ensure_loaded()
    return _cache.get(slug)


def count() -> int:
    """Kütüphanedeki analiz sayısı."""
    ensure_loaded()
    return len(_cache)


def video_path(slug: str) -> Optional[Path]:
    """Analize ait video dosyasının tam yolunu döner.

    Dosya adları emoji ve Latin dışı karakterler içerdiği için doğrudan URL'de
    taşınmaz; arayüz slug kullanır, gerçek yol burada çözülür.
    """
    data = get(slug)
    if not data:
        return None
    path = Path(data["_video_abs"])
    return path if path.exists() else None


def event_timestamps(slug: str) -> List[dict]:
    """Analizin uyarı zamanlama listesini döner (replay motoru kullanır)."""
    data = get(slug)
    if not data:
        return []
    return data.get("metadata", {}).get("event_timestamps", []) or []


def event_window(event: dict) -> tuple[float, float]:
    """Bir olayın videoda görünür olacağı ``[baslangic, bitis]`` aralığını döner.

    Karar ajanı ``timestamp_sec`` (mutlak başlangıç saniyesi) ve ``duration``
    (saniye) üretir (bkz. ``src/output/schema.py``). Model süre üretmediyse
    (``duration == 0``) aralık sıfır genişlikte döner; saha ekranı bu durumda
    klip stop mantığını uygulamaz, video sonuna kadar oynar — süre
    uydurulmaz.

    Args:
        event: ``metadata.event_timestamps`` içindeki bir olay sözlüğü.

    Returns:
        ``(baslangic_sec, bitis_sec)`` ikilisi.
    """
    start = float(event.get("timestamp_sec") or event.get("seconds") or 0.0)
    duration = float(event.get("duration") or 0.0)
    return start, start + max(0.0, duration)


def pick_event(analysis: dict, event_index: Optional[int] = None) -> Dict[str, Any]:
    """Atama için ilgili olayı seçer.

    ``event_index`` verilirse o olay, verilmezse **en kritik** olay seçilir.
    Kritiklik ölçütü sırasıyla şiddet (severity) ve güven skorudur; süpervizör
    bir olay belirtmediğinde ekibe en acil olanın gitmesi beklenir.

    Args:
        analysis: Analiz sözlüğü.
        event_index: ``metadata.event_timestamps`` içindeki indeks.

    Returns:
        Seçilen olay sözlüğü; hiç olay yoksa boş sözlük.
    """
    stamps = analysis.get("metadata", {}).get("event_timestamps", []) or []
    if not stamps:
        return {}

    if event_index is not None and 0 <= event_index < len(stamps):
        return stamps[event_index]

    severity_rank = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    return max(
        stamps,
        key=lambda e: (
            severity_rank.get(str(e.get("severity") or "low"), 1),
            float(e.get("confidence") or 0.0),
        ),
    )


def public_view(analysis: dict) -> dict:
    """Analizin arayüze gönderilecek biçimini üretir.

    Ham analiz dosyası tanı amaçlı büyük alanlar içerir (``rag_context``,
    ``vlm_interpretation``, kare indeksleri). Bunlar arayüzde kullanılmadığı
    için ağ üzerinden taşınmaz; yalnızca gösterim için gerekenler döner.
    """
    meta = analysis.get("metadata", {})
    return {
        "slug": analysis.get("slug"),
        "video_name": analysis.get("video_name"),
        "video": analysis.get("video", {}),
        "risk": analysis.get("risk"),
        "overall_risk": analysis.get("overall_risk", "unknown"),
        "scene_context": analysis.get("scene_context", {}),
        "results": analysis.get("results", []),
        "uncertain": analysis.get("uncertain", False),
        "uncertainty_reason": analysis.get("uncertainty_reason", ""),
        "confidence": analysis.get("confidence"),
        "headline": analysis.get("headline"),
        "summary": analysis.get("summary"),
        "reasoning": analysis.get("reasoning"),
        "actions": analysis.get("actions", []),
        "events": analysis.get("events", []),
        "event_timestamps": meta.get("event_timestamps", []),
        "triggered_mock_tools": analysis.get("triggered_mock_tools", []),
        "segment_count": meta.get("segment_count", 1),
        "channel_b_mode": meta.get("channel_b_mode", ""),
        "analyzed_at": analysis.get("analyzed_at"),
    }
