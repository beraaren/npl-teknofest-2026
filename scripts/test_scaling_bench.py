#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Kanal_B"))

from Kanal_B.preprocessing import detect_scene_boundaries, extract_video_segment, probe_video

videos_to_test = [
    "Normal_Videos_345_x264.mp4",     # 320x240 (7s)
    "3_te1.mp4",                     # 1080p (10s)
    "44889-440261137_medium.mp4",    # 1080p (16s)
    "41501-429661287_medium.mp4",    # 2560x1440 2K (27s)
]

print("=" * 80)
print("🔬 ÖLÇEKLENDİRME, SAHNE TESPİTİ VE CPU/GPU ETKİSİ DETAYLI BENCHMARK")
print("=" * 80)

for vname in videos_to_test:
    vpath = REPO_ROOT / "videos" / vname
    if not vpath.exists():
        continue
    
    info = probe_video(str(vpath))
    size_mb = vpath.stat().st_size / (1024 * 1024)
    w, h = info["width"], info["height"]
    dur = info["duration_sec"]
    
    print(f"\n🎥 Video: {vname}")
    print(f"   Boyut: {w}x{h} ({dur:.1f} sn, {size_mb:.2f} MB)")
    
    # 1. Sahne Tespiti (CPU)
    t0 = time.perf_counter()
    ranges = detect_scene_boundaries(str(vpath), min_segment_sec=5.0, max_segment_sec=60.0)
    t_scene = time.perf_counter() - t0
    print(f"   [CPU] 1. Sahne Sınırı Tespiti (SSIM / Akış): {t_scene:.3f} sn ({len(ranges)} segment)")
    
    # 2. 720p Ölçeklendirme ve Re-encode (libx264 CPU)
    t0 = time.perf_counter()
    out_test = str(REPO_ROOT / "scratch" / f"bench_{vpath.stem}_720p.mp4")
    os.makedirs(os.path.dirname(out_test), exist_ok=True)
    extract_video_segment(str(vpath), ranges[0][0], ranges[0][1], out_test, max_height=720)
    t_scale = time.perf_counter() - t0
    out_size_mb = os.path.getsize(out_test) / (1024 * 1024)
    print(f"   [CPU] 2. 720p Ölçekleme & Yeniden Kodlama (PyAV libx264): {t_scale:.3f} sn (Çıktı: {out_size_mb:.2f} MB)")
    
    # 3. Base64 Encode süresi
    t0 = time.perf_counter()
    import base64
    with open(out_test, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    t_b64 = time.perf_counter() - t0
    print(f"   [CPU/RAM] 3. Base64 Kodlama: {t_b64:.3f} sn (Base64 boyutu: {len(b64)/(1024*1024):.2f} MB)")

print("\n" + "=" * 80)
