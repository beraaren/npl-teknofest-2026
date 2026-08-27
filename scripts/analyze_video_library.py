#!/usr/bin/env python3
"""Video kütüphanesini toplu analiz eder ve sözde-canlı (pseudo-live) veri üretir.

NE YAPAR
--------
``videos/`` klasöründeki her videoyu, projenin tam analiz hattından geçirir
(Kanal A: YOLO + tracker + kural motoru → RAG; Kanal B: EVREN ``vlm`` modeliyle
bağımsız video yorumu; birleşim: Karar Ajanı → Guardrail → mock araçlar) ve
sonucu ``data/library/analyses/<slug>.json`` olarak kaydeder.

NEDEN ÖNCEDEN (OFFLINE) ANALİZ
------------------------------
Arayüz canlı bir kamera sistemi gibi davranır ama çalışma anında **hiçbir model
çağrısı yapmaz**. Analiz bir kez burada yapılır; arayüz yalnızca kaydedilmiş
sonuçları videonun oynatma konumuyla eşzamanlı göstererek canlı izlenimi verir.
Bunun üç faydası var: gösterim sırasında ağ/model gecikmesi yok, sonuçlar
tekrarlanabilir (aynı video her zaman aynı uyarıyı üretir) ve ortak kullanılan
çıkarım servisine gereksiz yük binmez.

ÜRETİLEN JSON'UN ARAYÜZ İÇİN KRİTİK ALANI
-----------------------------------------
``metadata.event_timestamps``: her olayın videonun başından itibaren **mutlak
saniyesi**. Replay motoru bu saniyeleri kullanarak uyarıyı tam olayın geçtiği
anda yayınlar; böylece uyarılar videoyla senkron görünür.

KULLANIM
--------
    python scripts/analyze_video_library.py                  # eksikleri analiz et
    python scripts/analyze_video_library.py --force          # hepsini yeniden analiz et
    python scripts/analyze_video_library.py --limit 3        # ilk 3 videoyu analiz et
    python scripts/analyze_video_library.py --only lathe     # adı eşleşenleri analiz et
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import traceback
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
# Kanal_B modülleri birbirini paket öneki olmadan içe aktarır; dizinin kendisi
# sys.path'te olmalı (test_akis.py ile aynı desen).
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "Kanal_B")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

# .env dosyasını burada, en başta yükle. Script bir IDE'nin "Çalıştır" (Run)
# düğmesiyle başlatıldığında terminale önceden EVREN_API_KEY export edilmiş
# olmayabilir; src.config.load_config() de .env'i yükler ama bu yalnızca ilk
# video işlenirken (analyze_video() içinde) çağrılır. Anahtarın script'in en
# başından itibaren hazır olması için burada da bir kez denenir.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env", override=False)
except ImportError:
    pass

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Karar ajanının olay tipinden kart başlığı üretmek için kullanılan sözlük.
# Başlık ayrı bir model çağrısıyla üretilmez: olay tipi + risk zaten karar
# ajanının doğrulanmış çıktısıdır, ek çağrı hem maliyet hem sapma riskidir.
EVENT_TITLES = {
    "forklift_tip_over": "Araç devrilmesi",
    "person_fall": "Personel düşmesi",
    "immobile_person": "Hareketsiz personel",
    "gathering": "Personel kalabalıklaşması",
    "ppe_missing": "KKD eksikliği",
    "dangerous_proximity": "Tehlikeli yakınlık",
    "fire_smoke": "Yangın / duman",
    "leakage": "Sızıntı",
    "pallet_collapse": "Palet çökmesi",
    "electrical_hazard": "Elektrik tehlikesi",
    "blocked_emergency_exit": "Acil çıkış engeli",
}

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}
RISK_RANK = {"Düşük": 1, "Orta": 2, "Yüksek": 3}


# ---------------------------------------------------------------------------
# Kimlik ve zaman yardımcıları
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    """Dosya adından URL ve dosya sistemi için güvenli bir kimlik üretir.

    Kütüphanedeki dosya adları emoji, Tamil karakterler, boşluk ve parantez
    içeriyor. Bunlar hem URL yolunda hem JSON dosya adında sorun çıkarır.
    Bu yüzden ad sadeleştirilir; ayrıca özgün adın kısa bir karma değeri eklenir
    ki sadeleştirme sonrası çakışan iki ad aynı kimliğe düşmesin.

    Args:
        name: Özgün dosya adı (uzantısız).

    Returns:
        ``a-z0-9-`` karakterlerinden oluşan, en fazla ~48 karakterlik kimlik.
    """
    turkish = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    text = name.translate(turkish)
    # Aksanları ayrıştırıp at (é -> e); emoji ve Latin dışı harfler düşer.
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    text = re.sub(r"-{2,}", "-", text)[:40].strip("-") or "video"
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:6]
    return f"{text}-{digest}"


def mmss(seconds: float) -> str:
    """Saniyeyi ``MM:SS`` biçimine çevirir."""
    total = max(0, int(round(seconds)))
    return f"{total // 60:02d}:{total % 60:02d}"


def time_to_seconds(value: Any) -> float:
    """``MM:SS`` / ``HH:MM:SS`` / sayı biçimindeki zamanı saniyeye çevirir.

    Karar ajanı zamanı ``MM:SS`` metni olarak yazar, kural motoru da öyle;
    Kanal B ise sayı üretir. Replay motoru saniye ile çalıştığı için burada
    tek biçime indirilir. Ayrıştırılamayan değer ``0.0`` olur.
    """
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    match = re.search(r"(\d{1,3}):(\d{1,2})(?::(\d{1,2}))?", str(value or ""))
    if not match:
        return 0.0
    if match.group(3) is not None:
        return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3))
    return int(match.group(1)) * 60 + int(match.group(2))


# ---------------------------------------------------------------------------
# Analiz sonucundan arayüz alanlarını türet
# ---------------------------------------------------------------------------

def build_headline(events: list[dict], risk: str) -> str:
    """Olay listesinden kısa bir kart başlığı üretir.

    En yüksek güvenli olay başlığı esas alınır; olay yoksa risk seviyesine
    dayalı genel bir başlık döner.
    """
    if not events:
        return f"{risk} riskli durum" if risk != "Düşük" else "Olağan operasyon"
    top = max(events, key=lambda e: float(e.get("confidence") or 0.0))
    event_type = str(top.get("event_type") or "").strip()
    title = EVENT_TITLES.get(event_type)
    if title:
        return f"{title} ({mmss(time_to_seconds(top.get('time')))})"
    text = str(top.get("event") or "Riskli durum").strip()
    return text[:70] + ("…" if len(text) > 70 else "")


def build_event_timestamps(
    final_output: dict,
    vlm_interpretation: dict,
    duration_sec: float,
) -> list[dict]:
    """Replay motorunun uyarı zamanlamasında kullanacağı olay listesini kurar.

    Karar ajanının doğrulanmış olayları esas alınır (kanıt birleştirmesi orada
    yapılmıştır). Şiddet (``severity``) bilgisi, zamanı en yakın olan Kanal B
    risk olayından ödünç alınır; Kanal B'de karşılığı yoksa risk seviyesinden
    türetilir.

    Zaman damgaları video süresine kırpılır: model bazen süre dışına taşan
    damga üretebiliyor ve bu, uyarının hiç tetiklenmemesine yol açar.

    Args:
        final_output: Guardrail'den geçmiş nihai karar çıktısı.
        vlm_interpretation: Kanal B (S8) yorumu.
        duration_sec: Videonun toplam süresi.

    Returns:
        ``seconds`` alanına göre sıralı olay sözlükleri.
    """
    vlm_events = [
        {
            "seconds": time_to_seconds(ev.get("timestamp_sec")),
            "severity": str(ev.get("severity") or "medium"),
            "description": str(ev.get("description_tr") or ""),
        }
        for ev in (vlm_interpretation or {}).get("risk_events", [])
        if isinstance(ev, dict)
    ]

    risk = str(final_output.get("risk") or "Düşük")
    default_severity = {"Yüksek": "high", "Orta": "medium", "Düşük": "low"}.get(risk, "medium")

    out: list[dict] = []
    for ev in final_output.get("events", []):
        if not isinstance(ev, dict):
            continue
        seconds = time_to_seconds(ev.get("time"))
        if duration_sec > 0:
            # Süreyi aşan damgayı sona çek; aksi hâlde uyarı hiç tetiklenmez.
            seconds = min(seconds, max(0.0, duration_sec - 0.5))

        # VLM risk olaylarından en yakın olayı bul; duration ve detay ödünç al.
        severity = default_severity
        detail = ""
        event_duration = 0.0
        if vlm_events:
            nearest = min(vlm_events, key=lambda v: abs(v["seconds"] - seconds))
            if abs(nearest["seconds"] - seconds) <= 5.0:
                severity = nearest["severity"]
                detail = nearest["description"]

        # Karar ajanı zaten duration/end_time/timestamp_sec üretmişse onları kullan;
        # yoksa VLM'den ödünç al veya varsayılan bırak.
        event_duration = float(ev.get("duration") or event_duration)
        timestamp_sec = float(ev.get("timestamp_sec") or seconds)
        end_time = str(ev.get("end_time") or "")
        if not end_time and event_duration > 0:
            end_time = mmss(timestamp_sec + event_duration)

        out.append({
            "seconds": round(timestamp_sec, 2),
            "timestamp": mmss(timestamp_sec),
            "end_time": end_time,
            "timestamp_sec": round(timestamp_sec, 2),
            "duration": round(event_duration, 2),
            "event_type": str(ev.get("event_type") or ""),
            "event": str(ev.get("event") or ""),
            "confidence": float(ev.get("confidence") or 0.0),
            "severity": severity,
            "vlm_detail": detail,
        })

    out.sort(key=lambda e: e["seconds"])
    return out


# ---------------------------------------------------------------------------
# Tek bir videoyu analiz et
# ---------------------------------------------------------------------------

def analyze_video(video_path: Path, out_dir: Path, config_path: str) -> dict:
    """Bir videoyu tam hattan geçirir ve arayüz için JSON üretir.

    Args:
        video_path: Analiz edilecek video.
        out_dir: JSON'un yazılacağı klasör.
        config_path: ``config.yaml`` yolu.

    Returns:
        Kaydedilen sonuç sözlüğü.
    """
    # Ağır bağımlılıklar fonksiyon içinde: script --help ile hızlı açılsın.
    import numpy as np
    from PIL import Image

    from preprocessing import probe_video  # Kanal_B/preprocessing.py
    from src.config import load_config
    from src.events.event_engine import EventEngine
    from src.models.vlm_backend import create_backend
    from src.output.guardrail import OutputGuardrail
    from src.perception.observer_agent import ObserverAgent
    from src.preprocessing.enhancer import LowLightEnhancer
    from src.preprocessing.frame_sampler import FrameSampler
    from src.preprocessing.video_reader import VideoReader
    from src.reasoning.decision_agent import DecisionAgent
    from src.reasoning.memory import ShortTermMemory
    from src.reasoning.mock_tools import MockToolRegistry
    from src.reasoning.rag_layer import RAGLayer

    config = load_config(config_path)
    slug = slugify(video_path.stem)
    started = time.time()

    info = probe_video(str(video_path))
    duration_sec = float(info["duration_sec"])

    # --- Kanal A & Kanal B Paralel Çalıştırma -------------------------
    from concurrent.futures import ThreadPoolExecutor
    from pipeline import run_channel_b

    channel_b_dir = out_dir.parent / "channel_b" / slug

    def _run_channel_a_worker():
        reader = VideoReader(str(video_path))
        native_fps = reader.fps or 25.0
        target_fps = config.preprocessing.channel_a_fps
        step = max(1, round(native_fps / target_fps)) if target_fps > 0 else 1
        fps_a = native_fps / step

        channel_a_frames, channel_a_indices = [], []
        for idx, frame in enumerate(reader.iter_frames()):
            if idx % step == 0:
                channel_a_frames.append(frame)
                channel_a_indices.append(idx)
        reader.close()

        if not channel_a_frames:
            raise ValueError("Videodan kare okunamadı.")

        observer = ObserverAgent(config.perception)
        observations = observer.observe_video(
            channel_a_frames, native_fps, sampled_indices=channel_a_indices
        )
        scene_graphs = [obs["scene_graph"] for obs in observations]

        engine = EventEngine(config.events, fps=fps_a)
        for obs in observations:
            engine.process_observation(obs)
        event_signals = engine.get_signals()

        rag = RAGLayer()
        rag_context = rag.build_context(observations, event_signals)
        memory = ShortTermMemory()
        for sig in event_signals:
            memory.add(sig, entry_type="event")

        return observations, scene_graphs, event_signals, rag_context, memory, fps_a, channel_a_indices, channel_a_frames, rag

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(_run_channel_a_worker)
        future_b = executor.submit(
            run_channel_b, str(video_path), video_id=slug, output_dir=str(channel_b_dir)
        )

        observations, scene_graphs, event_signals, rag_context, memory, fps_a, channel_a_indices, channel_a_frames, rag = future_a.result()
        vlm_interpretation = future_b.result()

    channel_b_mode = "video"
    tools = MockToolRegistry()

    # --- Karar Ajanı + LLM Çıkarımı -----------------------------------
    backend = create_backend(config.vlm, force="server")
    agent = DecisionAgent(
        config=config.decision_agent, vlm_config=config.vlm,
        rag=rag, memory=memory, tools=tools, backend=backend,
    )

    # --- Karar + guardrail --------------------------------------------
    decision_raw = agent.decide(
        event_signals=event_signals,
        scene_graphs=scene_graphs,
        rag_context=rag_context,
        vlm_interpretation=vlm_interpretation,
    )
    guardrail = OutputGuardrail(config.output.guardrail)
    final_output = guardrail.validate(
        decision_raw["raw_text"],
        decision_raw["retry_fn"],
        rag_risk_level=decision_raw["rag_risk_level"],
    )

    # --- Mock araçları çalıştır ---------------------------------------
    triggered = final_output.get("triggered_mock_tools") or []
    if not triggered:
        suggested = tools.suggest_tools(
            final_output["risk"],
            [e.get("event_type", "") for e in final_output.get("events", [])],
        )
        triggered = [
            {
                "tool_name": s["tool_name"],
                "params": {"location": "saha", "reason": final_output["summary"][:100]},
            }
            for s in suggested
        ]
        final_output["triggered_mock_tools"] = triggered

    final_output["tool_execution_results"] = [
        tools.execute(call["tool_name"], call.get("params", {})) for call in triggered
    ]

    # --- Arayüz alanlarını türet --------------------------------------
    events = final_output.get("events") or []
    event_timestamps = build_event_timestamps(final_output, vlm_interpretation, duration_sec)

    result = {
        "slug": slug,
        "video_file": video_path.relative_to(REPO_ROOT).as_posix(),
        "video_name": video_path.stem,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "analysis_seconds": round(time.time() - started, 1),
        "video": {
            "duration_sec": round(duration_sec, 2),
            "fps": round(float(info["fps"]), 2),
            "width": int(info["width"]),
            "height": int(info["height"]),
        },
        # Karar ajanının yazdığı olay özeti — süpervizör ve saha ekibi aynı
        # metni görür (tek doğruluk kaynağı).
        "summary": final_output.get("summary", ""),
        "headline": build_headline(events, str(final_output.get("risk") or "Düşük")),
        "risk": final_output.get("risk", "Düşük"),
        "confidence": final_output.get("confidence", 0.0),
        "reasoning": final_output.get("reasoning", ""),
        "actions": final_output.get("actions", []),
        "events": events,
        "triggered_mock_tools": final_output.get("triggered_mock_tools", []),
        "tool_execution_results": final_output.get("tool_execution_results", []),
        "metadata": {
            "vlm_backend": backend.name(),
            "channel_b_mode": channel_b_mode,
            "segment_count": vlm_interpretation.get("segment_count", 1),
            "failed_segments": vlm_interpretation.get("failed_segments", []),
            "channel_a_fps": round(fps_a, 2),
            "total_frames": len(channel_a_frames),
            "sampled_indices": channel_a_indices,
            "geometric_signals": event_signals,
            "vlm_interpretation": vlm_interpretation,
            "rag_context": rag_context,
            # Replay motorunun uyarı zamanlamasında kullandığı alan.
            "event_timestamps": event_timestamps,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{slug}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ---------------------------------------------------------------------------
# Toplu çalıştırma
# ---------------------------------------------------------------------------

def discover_videos(video_dir: Path) -> list[Path]:
    """Klasördeki video dosyalarını ada göre sıralı döner."""
    return sorted(
        p for p in video_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-dir", default=str(REPO_ROOT / "videos"), help="Video klasörü")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "library" / "analyses"),
                        help="JSON çıktı klasörü")
    parser.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    parser.add_argument("--force", action="store_true", help="Var olan analizleri de yeniden üret")
    parser.add_argument("--limit", type=int, default=0, help="En fazla bu kadar video analiz et")
    parser.add_argument("--only", default="", help="Adında bu metin geçen videoları analiz et")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    out_dir = Path(args.out_dir)

    if not video_dir.exists():
        print(f"HATA: video klasörü bulunamadı: {video_dir}")
        return 1

    videos = discover_videos(video_dir)
    if args.only:
        needle = args.only.lower()
        videos = [v for v in videos if needle in v.name.lower()]
    if not videos:
        print(f"HATA: {video_dir} içinde video bulunamadı.")
        return 1

    pending: list[Path] = []
    for video in videos:
        target = out_dir / f"{slugify(video.stem)}.json"
        if target.exists() and not args.force:
            print(f"— atlandı (analiz var): {video.name}")
            continue
        pending.append(video)

    if args.limit > 0:
        pending = pending[: args.limit]

    if not pending:
        print("Tüm videolar zaten analiz edilmiş. Yeniden üretmek için --force kullan.")
        return 0

    print(f"{len(pending)} video analiz edilecek (toplam {len(videos)} video bulundu).\n")

    succeeded: list[dict] = []
    failed: list[dict] = []

    for i, video in enumerate(pending, start=1):
        print(f"[{i}/{len(pending)}] {video.name}")
        try:
            result = analyze_video(video, out_dir, args.config)
            succeeded.append(result)
            print(
                f"    ✓ risk={result['risk']} güven={result['confidence']} "
                f"olay={len(result['events'])} damga={len(result['metadata']['event_timestamps'])} "
                f"segment={result['metadata']['segment_count']} "
                f"({result['analysis_seconds']}s)"
            )
            print(f"      {result['headline']} — {result['summary'][:110]}")
        except Exception as exc:
            failed.append({"video": video.name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"    ✗ BAŞARISIZ: {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)

    # Manifest: arayüz hangi analizlerin hazır olduğunu buradan öğrenir.
    manifest_path = out_dir.parent / "manifest.json"
    all_analyses = sorted(out_dir.glob("*.json"))
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(all_analyses),
        "analyses": [p.stem for p in all_analyses],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 68}")
    print(f"Başarılı: {len(succeeded)} | Başarısız: {len(failed)}")
    print(f"Kütüphanedeki toplam analiz: {manifest['count']}")
    print(f"Manifest: {manifest_path}")
    for item in failed:
        print(f"  ✗ {item['video']}: {item['error']}")
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
