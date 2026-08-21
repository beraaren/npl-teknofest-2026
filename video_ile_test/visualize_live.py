"""TEKNOFEST 2026 Senaryo 3 — Canlı Video Analiz ve Görselleştirici (HUD & Scene Graph).

Bu script, test videolarını kare kare oynatarak:
  1. YOLO-World tespitlerini ve ByteTrack takip ID'lerini (Bounding Box & Hareket İzi),
  2. Anlık Sahne Grafiği ilişkilerini (Yakınlık, KKD Giyme, Palet Taşıma çizgilerini),
  3. Kural Motorunun ürettiği anlık İSG uyarılarını (Kritik/Tehlike/Uyarı HUD banner'ını)
video üzerinde gerçek zamanlı olarak görselleştirir.

Kullanım:
  python test_videos/visualize_live.py --video tip_over.mp4
  python test_videos/visualize_live.py --video proximity.mp4 --save
  python test_videos/visualize_live.py --video ppe.mp4
  python test_videos/visualize_live.py --video human_fall.mp4
  python test_videos/visualize_live.py --video gathering.mp4

Klavye Kısayolları:
  [SPACE] : Duraklat / Devam Et
  [D]     : Sahne Grafiği Çizgilerini Aç / Kapat
  [T]     : Hareket İzlerini (Trails) Aç / Kapat
  [Q/ESC] : Çıkış
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# Proje ana dizinini sys.path'e ekle
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.events.event_engine import EventEngine
from src.perception.observer_agent import ObserverAgent


# Renk Paleti (BGR formatında)
CLASS_COLORS = {
    "arac": (255, 140, 0),      # Turuncu / Mavi-Turuncu
    "insan": (0, 255, 127),     # Canlı Zümrüt Yeşili
    "palet": (0, 215, 255),     # Sarı / Altın
    "baret": (255, 191, 0),     # Açık Mavi
    "yelek": (50, 205, 50),     # Fosforlu Yeşil
    "yangin": (0, 0, 255),      # Kırmızı
    "duman": (180, 180, 180),   # Gri
    "unknown": (200, 200, 200),
}

RELATION_COLORS = {
    "near": (0, 70, 255),       # Kırmızı-Turuncu (Tehlike)
    "wearing": (0, 255, 0),     # Yeşil (Güvenli KKD)
    "carrying": (255, 255, 0),   # Cyan (Taşıma)
}


def draw_overlay(frame: np.ndarray, alpha: float, x1: int, y1: int, x2: int, y2: int, color=(20, 20, 20)) -> None:
    """Yarı saydam cam (glassmorphism) arka plan çizer."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    sub = frame[y1:y2, x1:x2]
    rect = np.full_like(sub, color, dtype=np.uint8)
    cv2.addWeighted(rect, alpha, sub, 1.0 - alpha, 0, sub)


def draw_hud(
    frame: np.ndarray,
    video_name: str,
    frame_idx: int,
    total_frames: int,
    fps: float,
    active_tracks: int,
    recent_signals: List[Dict[str, Any]],
) -> None:
    """Üst bilgi çubuğu (HUD) ve aktif İSG uyarılarını çizer."""
    w = frame.shape[1]
    
    # 1. Üst Ana Başlık Barı
    draw_overlay(frame, 0.75, 0, 0, w, 55, color=(15, 15, 20))
    cv2.line(frame, (0, 55), (w, 55), (0, 200, 255), 2)

    # Sistem Bilgileri
    title = f"DALGA AI | ISG CANLI ANALIZ — [{video_name}]"
    stats = f"Kare: {frame_idx}/{total_frames} | FPS: {fps:.1f} | Aktif Nesne: {active_tracks}"
    cv2.putText(frame, title, (15, 25), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, stats, (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

    # Tuş Rehberi (Sağ Üst)
    guide = "[SPACE]: Duraklat | [D]: Graf Cizgileri | [Q]: Cikis"
    cv2.putText(frame, guide, (w - 380, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    # 2. Sol Alt Uyarı Paneli (Varsa alarmları göster)
    if recent_signals:
        panel_h = min(160, 30 + len(recent_signals) * 26)
        y_start = frame.shape[0] - panel_h - 10
        draw_overlay(frame, 0.85, 10, y_start, min(w - 10, 680), frame.shape[0] - 10, color=(10, 10, 30))
        cv2.rectangle(frame, (10, y_start), (min(w - 10, 680), frame.shape[0] - 10), (0, 0, 220), 2)
        
        cv2.putText(frame, "CANLI ISG KURAL VE TEHLIKE ALARMLARI", (20, y_start + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 100, 255), 1, cv2.LINE_AA)
        
        for idx, sig in enumerate(reversed(recent_signals[-4:])):
            y_pos = y_start + 45 + idx * 26
            event_type = sig.get("event_type", "UYARI")
            desc = sig.get("description", "")
            if len(desc) > 65:
                desc = desc[:62] + "..."
            
            # Alarm rozeti
            cv2.circle(frame, (25, y_pos - 4), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"[{event_type.upper()}] {desc}", (38, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)


def run_visualizer(video_path: Path, config_path: str = "config.yaml", save_output: bool = False, stride: int = 1, conf: float = 0.25) -> None:
    """Belirtilen videoyu canlı analiz penceresinde çalıştırır."""
    if not video_path.exists():
        print(f"❌ HATA: Video dosyası bulunamadı: {video_path}")
        return

    print(f"🎬 Video Analizi Başlatılıyor: {video_path.name}")
    cfg = load_config(config_path)
    if conf is not None:
        cfg.perception.confidence_threshold = conf

    # Video Yakalayıcı
    cap = cv2.VideoCapture(str(video_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 100)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, stride)
    effective_fps = native_fps / step

    # Gözlemci ve Kural Motoru
    observer = ObserverAgent(cfg.perception)
    engine = EventEngine(cfg.events, fps=effective_fps)

    # Video Kaydedici (İsteğe bağlı)
    out_writer = None
    if save_output:
        out_path = video_path.parent / f"annotated_{video_path.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(out_path), fourcc, native_fps, (width, height))
        print(f"💾 İşlenmiş video kaydedilecek: {out_path}")

    # Takip Geçmişi / Yörünge (Trails)
    track_trails: Dict[int, List[Tuple[int, int]]] = {}
    active_alerts: List[Dict[str, Any]] = []

    show_graph = True
    show_trails = True
    paused = False

    window_name = f"TEKNOFEST ISG Analiz - {video_path.name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, width), min(720, height))

    frame_idx = 0
    t_prev = time.time()

    while cap.isOpened():
        if not paused:
            if step > 1:
                for _ in range(step - 1):
                    if not cap.grab():
                        break
                    frame_idx += 1
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            timestamp = frame_idx / native_fps

            # 1. Algı ve Takip (ObserverAgent + ByteTrack)
            obs = observer.observe_frame(frame, frame_idx=frame_idx, timestamp=timestamp)

            # 2. Olay Motoru Değerlendirmesi
            new_signals = engine.process_observation(obs)
            for s in new_signals:
                active_alerts.append({
                    "event_type": s.event_type,
                    "description": s.description,
                    "involved": s.involved_track_ids,
                    "timestamp": timestamp,
                    "expire_frame": frame_idx + int(native_fps * 3), # 3 saniye ekranda tut
                })

            # Süresi dolan uyarıları temizle
            active_alerts = [a for a in active_alerts if a["expire_frame"] > frame_idx]

        # Görselleştirme Karesi
        display_frame = frame.copy()

        # Düğümlerin merkez koordinat haritası (Sahne grafiği çizgileri için)
        node_centers: Dict[str, Tuple[int, int]] = {}

        # 3. Nesne Kutuları, Etiketleri ve Yörüngeleri Çiz
        for det in obs.get("detections", []):
            cls_name = det.get("class", "unknown")
            tid = det.get("track_id")
            bbox = det.get("bbox", [0, 0, 0, 0])
            conf = det.get("confidence", 0.0)

            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = CLASS_COLORS.get(cls_name, (0, 255, 255))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Düğüm adını kaydet
            node_key = f"{cls_name}_{tid}" if tid is not None else f"{cls_name}_{cx}"
            node_centers[node_key] = (cx, cy)

            # Bounding Box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # Yörünge (Trail)
            if tid is not None:
                if tid not in track_trails:
                    track_trails[tid] = []
                if not paused:
                    track_trails[tid].append((cx, cy))
                    if len(track_trails[tid]) > 25:
                        track_trails[tid].pop(0)

                if show_trails and len(track_trails[tid]) > 1:
                    pts = np.array(track_trails[tid], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display_frame, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

            # Başlık Rozeti
            label = f"{cls_name.upper()} #{tid}" if tid is not None else f"{cls_name.upper()}"
            label += f" {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            
            draw_overlay(display_frame, 0.8, x1, max(0, y1 - th - 8), x1 + tw + 8, y1, color=color)
            cv2.putText(display_frame, label, (x1 + 4, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Sahne Grafiği İlişki Çizgileri (Edges)
        if show_graph:
            for edge in obs.get("scene_graph", {}).get("edges", []):
                src = edge.get("source")
                tgt = edge.get("target")
                rel = edge.get("relation", "near")

                pt1 = node_centers.get(src)
                pt2 = node_centers.get(tgt)

                if pt1 and pt2:
                    edge_color = RELATION_COLORS.get(rel, (0, 200, 255))
                    cv2.line(display_frame, pt1, pt2, edge_color, 2, cv2.LINE_AA)
                    
                    # Çizginin ortasına ilişki türünü yaz
                    mx, my = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
                    cv2.circle(display_frame, (mx, my), 4, edge_color, -1)
                    cv2.putText(display_frame, rel.upper(), (mx + 6, my - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, edge_color, 1, cv2.LINE_AA)

        # 5. Üst Bilgi ve İSG Alarm HUD'ı
        fps_calc = 1.0 / max(0.001, time.time() - t_prev)
        t_prev = time.time()
        active_count = len(obs.get("tracks", []))
        draw_hud(display_frame, video_path.name, frame_idx, total_frames, fps_calc, active_count, active_alerts)

        if out_writer:
            out_writer.write(display_frame)

        cv2.imshow(window_name, display_frame)

        # Klavye Kontrolleri
        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key in (ord('q'), 27):  # 'q' veya ESC
            break
        elif key == ord(' '):      # SPACE
            paused = not paused
        elif key in (ord('d'), ord('D')):
            show_graph = not show_graph
        elif key in (ord('t'), ord('T')):
            show_trails = not show_trails

    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Video Oynatımı Tamamlandı: {video_path.name}")


def main():
    parser = argparse.ArgumentParser(description="TEKNOFEST 2026 Canlı Video Analiz ve Sahne Grafiği Görselleştirici")
    parser.add_argument("--video", type=str, default="proximity.mp4", help="Analiz edilecek video adı (test_videos/ altında)")
    parser.add_argument("--config", type=str, default="config.yaml", help="config.yaml yolu")
    parser.add_argument("--stride", type=int, default=1, help="Kare atlama adımı (1 = her kare, 2 = 2 karede bir)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO güven eşiği (Varsayılan: 0.25)")
    parser.add_argument("--save", action="store_true", help="Görselleştirilmiş videoyu kaydet")
    args = parser.parse_args()

    video_dir = Path(__file__).resolve().parent
    video_path = video_dir / args.video if not Path(args.video).is_absolute() else Path(args.video)

    cfg = load_config(args.config)
    if args.conf:
        cfg.perception.confidence_threshold = args.conf

    run_visualizer(video_path, config_path=args.config, save_output=args.save, stride=args.stride, conf=args.conf)


if __name__ == "__main__":
    main()
