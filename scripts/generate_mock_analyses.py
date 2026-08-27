#!/usr/bin/env python3
"""Pseudo-live demo için 27 adet mock analiz JSON'u + placeholder video üretir.

Çıktılar:
  data/pseudolive/analyses/analysis_01.json ... analysis_27.json
  data/pseudolive/videos/video_01.mp4      ... video_27.mp4   (cv2 varsa)

JSON şeması, outputs/analysis_result.json (DecisionFinal + metadata) ile uyumludur;
UI/replay için ek alanlar metadata altındadır:
  video_file, camera_label, duration_sec, risk_segments[{start_sec,end_sec,event_type,risk,description}]

Kullanıcı gerçek analiz JSON'ları ve videolarla aynı dosya adlarını kullanarak
bunları birebir değiştirebilir (video_01.mp4 ↔ analysis_01.json).

Kullanım:  python scripts/generate_mock_analyses.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT_ANALYSES = DATA / "pseudolive" / "analyses"
OUT_VIDEOS = DATA / "pseudolive" / "videos"

VIDEO_COUNT = 27
FPS = 12  # placeholder video fps (JSON'daki duration ile uyumlu)

CAMERA_LABELS = [
    "Depo-A Giriş", "Depo-B Raf Arası", "Üretim Hattı 1", "Üretim Hattı 2",
    "Sevkiyat Rampası", "Forklift Şarj İstasyonu", "Palet İstif Alanı",
    "Ana Koridor", "Kantin Çıkışı", "Acil Çıkış Kapısı 3",
    "Kimyasal Depo", "Elektrik Panosu Odası",
]

RISK_ORDER = {"Düşük": 1, "Orta": 2, "Yüksek": 3}


def _mm_ss(seconds: int) -> str:
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _load_yaml(name: str) -> dict:
    with open(DATA / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _pick_actions(catalog: dict, risk_level: str, pattern_names: list[str]) -> list[str]:
    """action_catalog.yaml'dan önce pattern'e özel, sonra default aksiyonları toplar."""
    level = catalog.get("actions", {}).get(risk_level, {})
    actions: list[str] = []
    for p in pattern_names:
        specific = level.get(p)
        if isinstance(specific, list):
            actions.extend(specific)
    default = level.get("default", [])
    if isinstance(default, list):
        actions.extend(default)
    # tekrarsız, ilk 4
    seen, unique = set(), []
    for a in actions:
        if a not in seen:
            seen.add(a)
            unique.append(a)
    return unique[:4]


def _pick_tools(patterns: dict, pattern_names: list[str], max_tools: int = 3) -> list[dict]:
    tools: list[dict] = []
    seen: set[str] = set()
    for p in pattern_names:
        for tool in patterns.get(p, {}).get("mock_tool_hints", []):
            if tool in seen:
                continue
            seen.add(tool)
            tools.append({
                "tool_name": tool,
                "params": {"location": "saha", "reason": patterns[p].get("description", p)[:100]},
            })
            if len(tools) >= max_tools:
                return tools
    return tools


def generate_analysis(idx: int, rng: random.Random, patterns: dict, catalog: dict) -> dict:
    pattern_names = list(patterns.keys())
    label = rng.choice(CAMERA_LABELS)
    duration = rng.randint(25, 45)

    # ~%25'i "Normal" (olaysız) video olsun
    n_events = rng.choice([0, 0, 1, 1, 2, 2, 3])
    chosen = rng.sample(pattern_names, k=min(n_events, len(pattern_names)))

    events, signals, timestamps, segments = [], [], [], []
    used_seconds: list[int] = []
    for pname in chosen:
        pdata = patterns[pname]
        # aynı saniyeye denk gelmesin
        sec = rng.randint(3, duration - 8)
        while any(abs(sec - s) < 3 for s in used_seconds):
            sec = rng.randint(3, duration - 8)
        used_seconds.append(sec)

        desc = pdata.get("description", pname)
        conf = round(rng.uniform(0.55, 0.95), 2)
        events.append({
            "time": _mm_ss(sec),
            "event": desc,
            "event_type": pname,
            "confidence": conf,
        })
        signals.append({
            "event_type": pname,
            "timestamp": _mm_ss(sec),
            "description": desc,
            "confidence": conf,
            "involved_track_ids": [],
            "metadata": {},
        })
        timestamps.append({"event_type": pname, "timestamp": _mm_ss(sec), "seconds": sec})
        segments.append({
            "start_sec": sec,
            "end_sec": min(sec + rng.randint(5, 9), duration),
            "event_type": pname,
            "risk": pdata.get("risk_level", "Orta"),
            "description": desc,
        })

    if chosen:
        risk = max((patterns[p].get("risk_level", "Düşük") for p in chosen),
                   key=lambda r: RISK_ORDER.get(r, 0))
        top = max(chosen, key=lambda p: patterns[p].get("risk_score", 0))
        summary = f"{label} bölgesinde {len(chosen)} olay tespit edildi; en kritik: {patterns[top].get('description', top)}"
        reasoning = (
            f"Geometrik sinyaller {', '.join(chosen)} pattern'leriyle eşleşti. "
            f"En yüksek risk skoru {patterns[top].get('risk_score', 0)} ({top}). "
            "RAG bağlamı ve olay zaman damgaları tutarlı."
        )
        actions = _pick_actions(catalog, risk, chosen)
        tools = _pick_tools(patterns, chosen)
        confidence = round(rng.uniform(0.6, 0.92), 2)
    else:
        risk = "Düşük"
        summary = f"{label} bölgesinde anlamlı bir risk tespit edilmedi; rutin operasyon."
        reasoning = "Olay motoru anlamlı sinyal üretmedi; sahnede yalnızca normal faaliyet gözlendi."
        actions = _pick_actions(catalog, "Düşük", [])
        tools = []
        confidence = round(rng.uniform(0.7, 0.95), 2)

    return {
        "summary": summary,
        "events": events,
        "risk": risk,
        "actions": actions,
        "reasoning": reasoning,
        "confidence": confidence,
        "triggered_mock_tools": tools,
        "metadata": {
            "video_file": f"video_{idx:02d}.mp4",
            "camera_label": label,
            "duration_sec": duration,
            "fps": float(FPS),
            "geometric_signals": signals,
            "event_timestamps": sorted(timestamps, key=lambda t: t["seconds"]),
            "risk_segments": sorted(segments, key=lambda s: s["start_sec"]),
        },
    }


def generate_placeholder_video(idx: int, analysis: dict) -> bool:
    """cv2 varsa basit placeholder mp4 üretir (kamera etiketi + zaman + riskli kesitte kırmızı çerçeve)."""
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        return _generate_placeholder_video_ffmpeg(analysis)
    return _generate_placeholder_video_cv2(idx, analysis)


def _generate_placeholder_video_cv2(idx: int, analysis: dict) -> bool:
    import cv2
    import numpy as np

    meta = analysis["metadata"]
    duration = meta["duration_sec"]
    w, h = 640, 360
    path = OUT_VIDEOS / meta["video_file"]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))

    rng = random.Random(idx * 997)
    rect_x, rect_v = 50, rng.choice([3, 4, 5])
    total = duration * FPS
    for f in range(total):
        t = f / FPS
        frame = np.full((h, w, 3), (35, 35, 40), dtype=np.uint8)
        rect_x += rect_v
        if rect_x > w - 120 or rect_x < 20:
            rect_v = -rect_v
        cv2.rectangle(frame, (rect_x, 180), (rect_x + 80, 260), (200, 170, 60), -1)
        cv2.putText(frame, f"{meta['camera_label']}  [PLACEHOLDER]", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        cv2.putText(frame, f"t = {int(t // 60):02d}:{int(t % 60):02d}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        for seg in meta["risk_segments"]:
            if seg["start_sec"] <= t <= seg["end_sec"]:
                cv2.rectangle(frame, (4, 4), (w - 4, h - 4), (0, 0, 255), 6)
                cv2.putText(frame, seg["event_type"], (20, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        writer.write(frame)
    writer.release()
    return True


def _generate_placeholder_video_ffmpeg(analysis: dict) -> bool:
    """ffmpeg ile düz renk + yazı placeholder video üretir (cv2 olmadığı durum)."""
    meta = analysis["metadata"]
    duration = meta["duration_sec"]
    path = OUT_VIDEOS / meta["video_file"]
    label = meta['camera_label'].replace("'", "'"*3)  # drawtext escape
    # Font yolunu bul
    font_candidates = [
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    ]
    fontfile = next((p for p in font_candidates if Path(p).exists()), None)

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x232328:s=640x360:d={duration}:r={FPS}",
        "-vf", (
            f"drawtext=fontfile={fontfile}:text='{label} [PLACEHOLDER]':x=20:y=20:"
            "fontsize=24:fontcolor=white,"
            f"drawtext=fontfile={fontfile}:text='SASAI Demo Video':x=20:y=320:"
            "fontsize=18:fontcolor=gray"
        ),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(path),
    ]
    if not fontfile:
        # drawtext'siz sade düz video
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=0x232328:s=640x360:d={duration}:r={FPS}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ]
    import subprocess
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=max(30, duration))
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main() -> None:
    rng = random.Random(2026)
    patterns = _load_yaml("risk_patterns.yaml").get("patterns", {})
    catalog = _load_yaml("action_catalog.yaml")
    if not patterns:
        raise SystemExit("risk_patterns.yaml okunamadı veya boş.")

    OUT_ANALYSES.mkdir(parents=True, exist_ok=True)
    OUT_VIDEOS.mkdir(parents=True, exist_ok=True)

    video_ok = 0
    for i in range(1, VIDEO_COUNT + 1):
        analysis = generate_analysis(i, rng, patterns, catalog)
        out = OUT_ANALYSES / f"analysis_{i:02d}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        if generate_placeholder_video(i, analysis):
            video_ok += 1

    (OUT_VIDEOS / "README.txt").write_text(
        "Bu klasördeki video_XX.mp4 dosyaları placeholder'dır.\n"
        "Gerçek videolarınızı AYNI DOSYA ADIYLA (video_01.mp4 ... video_27.mp4) buraya koyun.\n"
        "Karşılık gelen analiz JSON'larını da data/pseudolive/analyses/analysis_XX.json olarak değiştirin.\n",
        encoding="utf-8",
    )

    print(f"{VIDEO_COUNT} analiz JSON'u → {OUT_ANALYSES}")
    print(f"{video_ok}/{VIDEO_COUNT} placeholder video → {OUT_VIDEOS}")


if __name__ == "__main__":
    main()
