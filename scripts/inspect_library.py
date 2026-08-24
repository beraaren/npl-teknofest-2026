#!/usr/bin/env python3
"""Üretilmiş kütüphane analizlerini özetleyerek doğrular.

``analyze_video_library.py`` çıktılarının arayüz için gereken alanları
(özellikle ``metadata.event_timestamps``) doğru taşıdığını hızlıca denetlemek
için kullanılır.

    python scripts/inspect_library.py            # özet tablo
    python scripts/inspect_library.py --detail   # olay damgalarıyla birlikte
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSES_DIR = REPO_ROOT / "data" / "library" / "analyses"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(ANALYSES_DIR))
    parser.add_argument("--detail", action="store_true", help="Olay damgalarını da yaz")
    args = parser.parse_args()

    files = sorted(Path(args.dir).glob("*.json"))
    if not files:
        print(f"Analiz bulunamadı: {args.dir}")
        return 1

    print(f"{len(files)} analiz bulundu: {args.dir}\n")
    header = f"{'slug':<48} {'risk':<7} {'gvn':>4} {'süre':>7} {'olay':>4} {'damga':>5} {'seg':>3} {'mod':<7}"
    print(header)
    print("-" * len(header))

    problems: list[str] = []
    total_events = 0

    for path in files:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            problems.append(f"{path.name}: okunamadı ({exc})")
            continue

        stamps = d.get("metadata", {}).get("event_timestamps", []) or []
        duration = float(d.get("video", {}).get("duration_sec") or 0)
        total_events += len(stamps)

        print(
            f"{d.get('slug', path.stem):<48} "
            f"{d.get('risk', '?'):<7} "
            f"{float(d.get('confidence') or 0):>4.2f} "
            f"{duration:>6.1f}s "
            f"{len(d.get('events') or []):>4} "
            f"{len(stamps):>5} "
            f"{d.get('metadata', {}).get('segment_count', '?'):>3} "
            f"{d.get('metadata', {}).get('channel_b_mode', '?'):<7}"
        )

        # Arayüzün çalışması için zorunlu alanlar
        if not str(d.get("summary") or "").strip():
            problems.append(f"{d.get('slug')}: summary BOŞ (süpervizör/saha ekranı boş kalır)")
        if not str(d.get("headline") or "").strip():
            problems.append(f"{d.get('slug')}: headline BOŞ")
        if not (d.get("actions") or []):
            problems.append(f"{d.get('slug')}: actions BOŞ (aksiyon önerileri çalışmaz)")
        if not Path(REPO_ROOT / d.get("video_file", "")).exists():
            problems.append(f"{d.get('slug')}: video dosyası yok -> {d.get('video_file')}")
        for stamp in stamps:
            secs = float(stamp.get("seconds") or 0)
            if duration > 0 and secs > duration:
                problems.append(
                    f"{d.get('slug')}: {stamp.get('timestamp')} damgası süreyi ({duration:.0f}s) aşıyor"
                )

        if args.detail and stamps:
            for stamp in stamps:
                detail = str(stamp.get("vlm_detail") or "")
                print(
                    f"    {stamp.get('timestamp')} ({float(stamp.get('seconds') or 0):6.1f}s) "
                    f"[{str(stamp.get('severity')):8}] {str(stamp.get('event_type')):22} "
                    f"gvn={float(stamp.get('confidence') or 0):.2f}"
                    + (f"  vlm: {detail[:70]}" if detail else "")
                )

    print(f"\nToplam uyarı damgası: {total_events}")
    if problems:
        print(f"\n{len(problems)} SORUN:")
        for p in problems:
            print(f"  ! {p}")
        return 2
    print("Tüm analizler arayüz için gereken alanları taşıyor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
