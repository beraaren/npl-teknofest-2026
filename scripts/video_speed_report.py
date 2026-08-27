#!/usr/bin/env python3
import json
from pathlib import Path

analyses_dir = Path('data/library/analyses')
records = []
for p in sorted(analyses_dir.glob('*.json')):
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        v_dur = float(data.get('video', {}).get('duration_sec') or 0.0)
        a_dur = float(data.get('analysis_seconds') or 0.0)
        w = data.get('video', {}).get('width', 0)
        h = data.get('video', {}).get('height', 0)
        fps = data.get('video', {}).get('fps', 0)
        segments = data.get('metadata', {}).get('segment_count', 1)
        frames = data.get('metadata', {}).get('total_frames', 0)
        name = data.get('video_name') or p.stem
        ratio = a_dur / v_dur if v_dur > 0 else 0
        records.append({
            'name': name[:38],
            'v_dur': v_dur,
            'a_dur': a_dur,
            'ratio': ratio,
            'res': f"{w}x{h}",
            'fps': fps,
            'segments': segments,
            'frames': frames
        })
    except Exception as e:
        print(f"Error reading {p.name}: {e}")

records.sort(key=lambda x: x['v_dur'])
print(f"{'#':<3} | {'Video Adı':<38} | {'Video (sn)':<10} | {'Analiz (sn)':<12} | {'Oran (Analiz/Video)':<20} | {'Çözünürlük':<12} | {'Segment'}")
print("-" * 115)
for i, r in enumerate(records, 1):
    print(f"{i:<3} | {r['name']:<38} | {r['v_dur']:>8.1f}s | {r['a_dur']:>10.1f}s | {r['ratio']:>18.2f}x | {r['res']:<12} | {r['segments']}")

short_vids = [r for r in records if r['v_dur'] <= 15]
mid_vids = [r for r in records if 15 < r['v_dur'] <= 60]
long_vids = [r for r in records if r['v_dur'] > 60]

print("\n" + "=" * 115)
print("📈 GRUP BAZLI ORTALAMALAR:")
print(f"  • Kısa Videolar (<= 15 sn, {len(short_vids)} adet): Ortalama Video: {sum(r['v_dur'] for r in short_vids)/len(short_vids):.1f}s -> Ortalama Analiz: {sum(r['a_dur'] for r in short_vids)/len(short_vids):.1f}s (Katsayı: {sum(r['ratio'] for r in short_vids)/len(short_vids):.2f}x)")
print(f"  • Orta Videolar (15-60 sn, {len(mid_vids)} adet): Ortalama Video: {sum(r['v_dur'] for r in mid_vids)/len(mid_vids):.1f}s -> Ortalama Analiz: {sum(r['a_dur'] for r in mid_vids)/len(mid_vids):.1f}s (Katsayı: {sum(r['ratio'] for r in mid_vids)/len(mid_vids):.2f}x)")
print(f"  • Uzun Videolar (> 60 sn, {len(long_vids)} adet): Ortalama Video: {sum(r['v_dur'] for r in long_vids)/len(long_vids):.1f}s -> Ortalama Analiz: {sum(r['a_dur'] for r in long_vids)/len(long_vids):.1f}s (Katsayı: {sum(r['ratio'] for r in long_vids)/len(long_vids):.2f}x)")
print("=" * 115)
