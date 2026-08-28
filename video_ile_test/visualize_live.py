"""TEKNOFEST 2026 Senaryo 3 — Canlı Video İSG Kural Denetim Görselleştirici.

Bu script, test videolarını kare kare oynatarak:
  1. Görüntünün üzerine doğrudan herhangi bir HUD / uyarı paneli bindirmeden,
  2. Alt kısma eklenen özel İSG Durum Şeridi (Dashboard) üzerinde tüm kuralları (6 kural)
     durum gösterge yuvarlakları ile (Normal: Kırmızı Yuvarlak, Tetiklendiğinde: Yeşil Yuvarlak),
  3. Tetikleme bittikten sonra belirlenen soğuma süresi (2.5 sn) boyunca yeşil kalıp
     ardından otomatik kırmızıya dönme mantığı ile canlı olarak görselleştirir.

Kural Listesi:
  1. FORKLIFT DEVRILME (`forklift_tip_over`): Yan yatma / denge kaybı.
  2. INSAN DUSMESI (`person_fall`): Ani dikey kinematik düşüş.
  3. TEHLIKELI YAKINLIK (`dangerous_proximity`): Araç ile yaya yakınlaşması.
  4. KKD DENETIMI (`ppe_missing`): Baret veya reflektör yelek eksikliği.
  5. TEHLIKELI TOPLANMA (`gathering`): Çoklu personel kümelenmesi.
  6. YANGIN / DUMAN (`fire_smoke`): Alev veya duman sınıfının süreklilik eşiğini aşarak tespiti.

Kullanım:
  1. Arayüz ile (Sürükle-Bırak veya Dosya Seç):
       python video_ile_test/visualize_live.py

  2. Doğrudan Terminalden Video Belirterek:
       python video_ile_test/visualize_live.py --video proximity.mp4
       python video_ile_test/visualize_live.py --video tip_over.mp4
       python video_ile_test/visualize_live.py --video ppe.mp4
       python video_ile_test/visualize_live.py --video human_fall.mp4
       python video_ile_test/visualize_live.py --video gathering.mp4

Klavye Kısayolları:
  [SPACE] : Duraklat / Devam Et
  [B]     : Nesne Kutularını (Bounding Box) Aç / Kapat
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
from typing import Any, Dict, List, Optional, Tuple

# Windows konsolu için UTF-8 çıktı desteği
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np

# Proje ana dizinini sys.path'e ekle
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.events.event_engine import EventEngine
from src.perception.observer_agent import ObserverAgent
from src.perception.scene_graph import PPE_CLASSES, SceneGraph


# =====================================================================
# RENK PALETİ VE KURAL TANIMLARI
# =====================================================================

CLASS_COLORS = {
    "arac": (255, 140, 0),      # Turuncu
    "insan": (0, 255, 127),     # Zümrüt Yeşili
    "palet": (0, 215, 255),     # Altın / Sarı
    "baret": (255, 191, 0),     # Açık Mavi
    "yelek": (50, 205, 50),     # Fosforlu Yeşil
    "yangin": (0, 0, 255),      # Kırmızı
    "duman": (180, 180, 180),   # Gri
    "unknown": (200, 200, 200),
}

RELATION_COLORS = {
    "near": (0, 70, 255),       # Kırmızı-Turuncu
    "wearing": (0, 255, 0),     # Yeşil
    "carrying": (255, 255, 0),  # Cyan
}

# Tetiklenen kuralın yeşil kaldıktan sonra kırmızıya dönmesi için geçecek yerleşik soğuma süresi (saniye)
ALERT_HOLD_DURATION_SECONDS: float = 2.5

# 6 Temel İSG Kuralı Tanımları
RULE_DEFINITIONS = [

    {
        "key": "forklift_tip_over",
        "title": "FORKLIFT DEVRILME",
        "subtitle": "Yan Yatma / Denge",
        "alias": ["forklift_tip_over", "tip_over"],
    },
    {
        "key": "person_fall",
        "title": "INSAN DUSMESI",
        "subtitle": "Ani Kinematik Dusme",
        "alias": ["person_fall", "fall"],
    },
    {
        "key": "dangerous_proximity",
        "title": "TEHLIKELI YAKINLIK",
        "subtitle": "Arac - Yaya Mesafe",
        "alias": ["dangerous_proximity", "proximity"],
    },
    {
        "key": "ppe_missing",
        "title": "KKD DENETIMI",
        "subtitle": "Baret / Yelek Eksik",
        "alias": ["ppe_missing", "ppe"],
    },
    {
        "key": "gathering",
        "title": "TEHLIKELI TOPLANMA",
        "subtitle": "Grup Kumelenme",
        "alias": ["gathering"],
    },
    {
        "key": "fire_smoke",
        "title": "YANGIN / DUMAN",
        "subtitle": "Alev / Duman Tespiti",
        "alias": ["fire_smoke", "fire", "smoke", "yangin", "duman"],
    },
]


def draw_overlay(frame: np.ndarray, alpha: float, x1: int, y1: int, x2: int, y2: int, color=(20, 20, 20)) -> None:
    """Yarı saydam dikdörtgen çizer."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    sub = frame[y1:y2, x1:x2]
    rect = np.full_like(sub, color, dtype=np.uint8)
    cv2.addWeighted(rect, alpha, sub, 1.0 - alpha, 0, sub)


def get_downloads_dir() -> Path:
    """Kullanıcının İndirilenler (Downloads) klasörünü bulur."""
    downloads = Path.home() / "Downloads"
    if not downloads.exists():
        downloads = Path.home() / "İndirilenler"
    downloads.mkdir(parents=True, exist_ok=True)
    return downloads


# =====================================================================
# ALT İSG DURUM ŞERİDİ (BOTTOM STATUS DASHBOARD)
# =====================================================================

def draw_bottom_panel(
    combined_frame: np.ndarray,
    y_offset: int,
    bar_height: int,
    video_name: str,
    frame_idx: int,
    total_frames: int,
    fps: float,
    active_tracks: int,
    rule_states: Dict[str, Dict[str, Any]],
    current_timestamp: float,
) -> None:
    """Görüntünün altına eklenen şeritte sistem durumunu ve 6 İSG kuralını çizer.

    Tetiklenmeyen durumlar: Kırmızı yuvarlak gösterge.
    Tetiklenen durumlar: Yeşil yuvarlak gösterge (2.5 sn soğuma süresi boyunca yeşil kalır).
    """
    total_w = combined_frame.shape[1]
    y_start = y_offset
    y_end = y_offset + bar_height

    # 1. Ana Panel Arka Planı (Koyu Slate / Lacivert Teması)
    combined_frame[y_start:y_end, 0:total_w] = (16, 18, 24)

    # 2. Üst Ayırıcı Çizgi (Cyan / Mavi Vurgu)
    cv2.line(combined_frame, (0, y_start), (total_w, y_start), (0, 180, 240), 2)

    # 3. Panel Üst Bilgi Başlığı (System Status Bar)
    # Sol Bilgi
    title_text = f"DALGA AI | ISG KURAL DENETIM SERITI  [{video_name}]"
    stats_text = f"Kare: {frame_idx}/{total_frames} | FPS: {fps:.1f} | Nesne: {active_tracks} | Zaman: {current_timestamp:.1f}s"
    cv2.putText(combined_frame, title_text, (14, y_start + 20), cv2.FONT_HERSHEY_DUPLEX, 0.44, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(combined_frame, stats_text, (14 + int(len(title_text) * 8.2) + 20, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 200, 210), 1, cv2.LINE_AA)

    # Sağ Kısayol Rehberi
    guide = "[SPACE]: Duraklat | [B]: Kutular | [D]: Graf | [T]: Izler | [Q]: Cikis"
    (gw, _), _ = cv2.getTextSize(guide, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
    cv2.putText(combined_frame, guide, (max(10, total_w - gw - 14), y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 150, 165), 1, cv2.LINE_AA)

    # İç İnce Ayırıcı Çizgi
    header_sep_y = y_start + 30
    cv2.line(combined_frame, (10, header_sep_y), (total_w - 10, header_sep_y), (35, 40, 52), 1)

    # 4. 6 Kural Kartlarının Çizimi
    num_rules = len(RULE_DEFINITIONS)
    is_wide_layout = (total_w >= 900)

    if is_wide_layout:
        # Tek satırda 6 kart
        margin_x = 10
        gap_x = 8
        avail_w = total_w - (2 * margin_x) - ((num_rules - 1) * gap_x)
        card_w = max(110, avail_w // num_rules)
        card_y1 = header_sep_y + 6
        card_y2 = y_end - 8

        for i, rule in enumerate(RULE_DEFINITIONS):
            r_key = rule["key"]
            state = rule_states.get(r_key, {"is_active": False, "time_left": 0.0, "detail": ""})
            is_active = state["is_active"]
            time_left = state["time_left"]

            cx1 = margin_x + i * (card_w + gap_x)
            cx2 = min(total_w - margin_x, cx1 + card_w)

            _draw_single_rule_card(
                combined_frame,
                cx1, card_y1, cx2, card_y2,
                rule=rule,
                is_active=is_active,
                time_left=time_left,
            )
    else:
        # Küçük çözünürlükler için responsive ızgara: satır sayısı kural sayısından
        # türetilir. Sabit 2 satır varsayımı, kural sayısı 6'yı aştığında (örn.
        # fire_smoke ile 7 kural) son satırın panel dışına taşmasına yol açardı.
        margin_x = 8
        gap_x = 6
        cols = 3
        rows_needed = (num_rules + cols - 1) // cols
        avail_w = total_w - (2 * margin_x) - ((cols - 1) * gap_x)
        card_w = max(90, avail_w // cols)
        row_h = max(34, (y_end - header_sep_y - 12) // rows_needed)

        for i, rule in enumerate(RULE_DEFINITIONS):
            r_key = rule["key"]
            state = rule_states.get(r_key, {"is_active": False, "time_left": 0.0, "detail": ""})
            is_active = state["is_active"]
            time_left = state["time_left"]

            row_idx = i // cols
            col_idx = i % cols

            cx1 = margin_x + col_idx * (card_w + gap_x)
            cx2 = min(total_w - margin_x, cx1 + card_w)
            cy1 = header_sep_y + 4 + row_idx * (row_h + 4)
            cy2 = cy1 + row_h

            _draw_single_rule_card(
                combined_frame,
                cx1, cy1, cx2, cy2,
                rule=rule,
                is_active=is_active,
                time_left=time_left,
                compact=True,
            )


def _draw_single_rule_card(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    rule: Dict[str, Any],
    is_active: bool,
    time_left: float,
    compact: bool = False,
) -> None:
    """Tek bir kural kartını ve durum gösterge yuvarlağını çizer."""
    # Kart Arka Planı ve Kenarlık
    if is_active:
        # Aktif / Tetiklendi: Canlı Koyu Yeşil Zemin + Parlak Yeşil Kenarlık
        bg_color = (12, 38, 20)
        border_color = (0, 225, 80)
        border_thick = 2
    else:
        # Pasif / Normal: Koyu Antrasit Zemin + İnce Gri Kenarlık
        bg_color = (24, 27, 34)
        border_color = (48, 54, 66)
        border_thick = 1

    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thick)

    card_h = y2 - y1
    card_w = x2 - x1

    # Üst Parıltı Çizgisi (Aktifken)
    if is_active:
        cv2.line(frame, (x1 + 1, y1 + 1), (x2 - 1, y1 + 1), (0, 255, 120), 2)

    # Durum Gösterge Yuvarlağı (Kırmızı: Pasif / Yeşil: Aktif)
    circle_x = x1 + 14
    circle_y = y1 + (18 if not compact else 15)

    if is_active:
        # YEŞİL YUVARLAK (Tetiklendi / Aktif Alarm)
        # Dış parıltı halkası
        cv2.circle(frame, (circle_x, circle_y), 9, (0, 160, 50), 1, cv2.LINE_AA)
        # Dolgulu canlı yeşil daire
        cv2.circle(frame, (circle_x, circle_y), 6, (50, 255, 50), -1, cv2.LINE_AA)
        # Merkez parlak nokta
        cv2.circle(frame, (circle_x, circle_y), 2, (220, 255, 220), -1, cv2.LINE_AA)
    else:
        # KIRMIZI YUVARLAK (Normal / Tetiklenmedi)
        # Dolgulu canlı kırmızı daire
        cv2.circle(frame, (circle_x, circle_y), 6, (0, 0, 220), -1, cv2.LINE_AA)
        # Dış koyu kırmızı kenarlık
        cv2.circle(frame, (circle_x, circle_y), 6, (0, 0, 110), 1, cv2.LINE_AA)

    # Başlık Metni
    title = rule["title"]
    font_title = cv2.FONT_HERSHEY_DUPLEX
    scale_title = 0.38 if card_w >= 160 else 0.32
    color_title = (255, 255, 255) if is_active else (205, 210, 220)
    cv2.putText(frame, title, (x1 + 26, circle_y + 4), font_title, scale_title, color_title, 1, cv2.LINE_AA)

    if not compact and card_h >= 65:
        # 2. Satır: Durum Rozeti
        state_y = y1 + 44
        if is_active:
            state_text = f"TETIKLENDI ({time_left:.1f}s)"
            state_color = (50, 255, 100)  # Parlak Yeşil
        else:
            state_text = "DURUM: NORMAL"
            state_color = (120, 135, 150)  # Gri

        cv2.putText(frame, state_text, (x1 + 10, state_y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, state_color, 1, cv2.LINE_AA)

        # 3. Satır: Alt Bilgi / Kapsam
        sub_y = y1 + 64
        sub_text = rule["subtitle"]
        sub_color = (140, 210, 160) if is_active else (95, 105, 120)
        cv2.putText(frame, sub_text, (x1 + 10, sub_y), cv2.FONT_HERSHEY_SIMPLEX, 0.30, sub_color, 1, cv2.LINE_AA)
    elif compact:
        # Kompakt Görünüm 2. Satır
        state_y = y1 + 34
        if is_active:
            state_text = f"TETIK ({time_left:.1f}s)"
            state_color = (50, 255, 100)
        else:
            state_text = "NORMAL"
            state_color = (120, 135, 150)
        cv2.putText(frame, state_text, (x1 + 10, state_y), cv2.FONT_HERSHEY_SIMPLEX, 0.30, state_color, 1, cv2.LINE_AA)


# =====================================================================
# H.264 VİDEO KAYDEDİCİ
# =====================================================================

class CompatibleVideoWriter:
    """WhatsApp, mobil cihazlar ve web tarayıcılarıyla tam uyumlu H.264 MP4 kaydedici."""

    def __init__(self, out_path: Path | str, fps: float, width: int, height: int):
        self.out_path = Path(out_path)
        self.use_av = False
        self.cv_writer = None

        # Boyutların çift sayı olmasını sağla
        w = width if width % 2 == 0 else width - 1
        h = height if height % 2 == 0 else height - 1
        self.width = w
        self.height = h

        try:
            import av
            self.container = av.open(str(self.out_path), mode="w")
            self.stream = self.container.add_stream("libx264", rate=max(1, int(round(fps))))
            self.stream.width = self.width
            self.stream.height = self.height
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"crf": "22", "preset": "veryfast"}
            self.use_av = True
        except Exception:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.cv_writer = cv2.VideoWriter(str(self.out_path), fourcc, fps, (self.width, self.height))

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        if self.use_av:
            import av
            av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            for packet in self.stream.encode(av_frame):
                self.container.mux(packet)
        elif self.cv_writer:
            self.cv_writer.write(frame)

    def release(self) -> None:
        if self.use_av:
            for packet in self.stream.encode():
                self.container.mux(packet)
            self.container.close()
        elif self.cv_writer:
            self.cv_writer.release()


# =====================================================================
# ANA ÇALIŞTIRICI (RUNNER)
# =====================================================================

def run_visualizer(
    video_path: Path,
    config_path: str = "config.yaml",
    save_output: bool = True,
    output_dir: Path | str | None = None,
    stride: int = 1,
    conf: float = 0.25,
    proximity_threshold: float | None = None,
    hold_sec: float = ALERT_HOLD_DURATION_SECONDS,
) -> None:
    """Belirtilen videoyu temiz görüntü ve alt İSG şeridi ile analiz eder."""
    if not video_path.exists():
        print(f"❌ HATA: Video dosyası bulunamadı: {video_path}")
        return

    print(f"🎬 Video Analizi Başlatılıyor: {video_path.name}")
    cfg = load_config(config_path)
    if conf is not None:
        cfg.perception.confidence_threshold = conf
    if proximity_threshold is not None:
        cfg.events.thresholds.proximity["distance_threshold_pixels"] = proximity_threshold
        print(f"📏 Yakınlık eşiği override edildi: {proximity_threshold:.0f}px")

    # Video Yakalayıcı
    cap = cv2.VideoCapture(str(video_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 100)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, stride)
    effective_fps = native_fps / step

    # Alt Şerit Boyutlandırması
    bar_height = 126 if width >= 900 else 156
    bar_height = bar_height if bar_height % 2 == 0 else bar_height + 1
    total_width = width if width % 2 == 0 else width - 1
    total_height = height + bar_height
    total_height = total_height if total_height % 2 == 0 else total_height - 1

    # Gözlemci ve Kural Motoru
    engine = EventEngine(cfg.events, fps=effective_fps)
    observer = ObserverAgent(cfg.perception, proximity_threshold=engine.proximity_threshold)

    # Video Kaydedici
    out_writer = None
    out_path = None
    if save_output:
        target_dir = Path(output_dir) if output_dir else get_downloads_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / f"annotated_{video_path.stem}.mp4"
        write_fps = effective_fps if step > 1 else native_fps
        out_writer = CompatibleVideoWriter(str(out_path), write_fps, total_width, total_height)
        print(f"💾 İşlenmiş video İndirilenler klasörüne kaydedilecek: {out_path}")

    # Takip ve Kural Zaman Geçmişi
    track_trails: Dict[int, List[Tuple[int, int]]] = {}
    last_trigger_timestamps: Dict[str, float] = {r["key"]: -999.0 for r in RULE_DEFINITIONS}
    last_trigger_details: Dict[str, str] = {r["key"]: "" for r in RULE_DEFINITIONS}

    show_boxes = True
    show_graph = False
    show_trails = False
    paused = False

    window_name = f"TEKNOFEST ISG Canli Kural Denetimi - {video_path.name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1280, total_width), min(760, total_height))

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

            # 2. Olay Motoru — TEK SEFER çalıştırılır. process_observation() zaten
            # kendi içinde _observation_to_tracks() ve states.update()'i çalıştırıp
            # track geçmişini (history) günceller. Bunu burada MANUEL olarak bir daha
            # yapıp sonra process_observation()'ı tekrar çağırmak, aynı karenin
            # tespitini history'ye İKİ KEZ eklerdi (aynı merkez arka arkaya) — bu da
            # speed_y'yi sahte şekilde sıfırlayıp state.update() içindeki fall_frames
            # sayacını her karede bir artırıp bir azaltarak asla 2'ye ulaşmasına izin
            # vermezdi (bkz. src/events/state_machine.py). person_fall bu yüzden
            # canlı görselleştiricide hiç tetiklenmiyordu.
            new_signals = engine.process_observation(obs)

            graph = SceneGraph.from_dict(
                obs.get("scene_graph", {}),
                proximity_threshold=engine.proximity_threshold,
            )

            # "Anlık" (dedup filtresiz) kural durumu için, process_observation()
            # tarafından ZATEN güncellenmiş kalıcı track nesnelerini ve durum
            # makinesini yeniden kullanıyoruz; tekrar üretmiyoruz.
            current_tracks = [
                engine._tracked_objects[t["track_id"]]
                for t in obs.get("tracks", [])
                if t["track_id"] in engine._tracked_objects
            ]
            raw_signals = engine.rules.evaluate(current_tracks, engine.states, graph)

            # Tetiklenen kuralların son aktif olma zamanını güncelle
            for sig in list(raw_signals) + list(new_signals):
                etype = sig.event_type
                for r in RULE_DEFINITIONS:
                    if etype == r["key"] or etype in r.get("alias", []):
                        last_trigger_timestamps[r["key"]] = timestamp
                        last_trigger_details[r["key"]] = sig.description

        # 3. Görüntü Katmanı — Görüntünün üzerine ağır HUD panelleri BİNDİRİLMEZ
        display_video = frame.copy()
        node_centers: Dict[str, Tuple[int, int]] = {}

        # İsteğe bağlı: Nesne kutuları (İnce ve temiz)
        if show_boxes:
            for det in obs.get("detections", []):
                cls_name = det.get("class", "unknown")
                tid = det.get("track_id")
                bbox = det.get("bbox", [0, 0, 0, 0])
                conf_val = det.get("confidence", 0.0)

                x1, y1, x2, y2 = [int(v) for v in bbox]
                color = CLASS_COLORS.get(cls_name, (0, 255, 255))
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                node_key = f"{cls_name}_{tid}" if tid is not None else f"{cls_name}_{cx}"
                node_centers[node_key] = (cx, cy)

                # İnce Bounding Box
                cv2.rectangle(display_video, (x1, y1), (x2, y2), color, 2)

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
                        cv2.polylines(display_video, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

                # Sade ve küçük etiket rozeti
                label = f"{cls_name.upper()} #{tid}" if tid is not None else f"{cls_name.upper()}"
                label += f" {conf_val:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
                draw_overlay(display_video, 0.75, x1, max(0, y1 - th - 6), x1 + tw + 6, y1, color=color)
                cv2.putText(display_video, label, (x1 + 3, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # İsteğe bağlı: Sahne Grafiği Çizgileri ([D] tuşuyla açılabilir)
        if show_graph:
            for edge in obs.get("scene_graph", {}).get("edges", []):
                src, tgt = edge.get("source"), edge.get("target")
                rel = edge.get("relation", "near")
                pt1, pt2 = node_centers.get(src), node_centers.get(tgt)
                if pt1 and pt2:
                    edge_color = RELATION_COLORS.get(rel, (0, 200, 255))
                    cv2.line(display_video, pt1, pt2, edge_color, 2, cv2.LINE_AA)
                    dist_px = math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
                    mx, my = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
                    cv2.circle(display_video, (mx, my), 4, edge_color, -1)
                    lbl = f"{rel.upper()} ({dist_px:.0f}px)" if rel in ("near", "carrying") else rel.upper()
                    cv2.putText(display_video, lbl, (mx + 6, my - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, edge_color, 1, cv2.LINE_AA)

        # 4. Birleşik Tuval (Composite Frame: Temiz Video + Alt İSG Şeridi)
        combined_frame = np.zeros((total_height, total_width, 3), dtype=np.uint8)

        # Üst kısmı video ile doldur
        if display_video.shape[1] != total_width or display_video.shape[0] != height:
            display_video = cv2.resize(display_video, (total_width, height))
        combined_frame[0:height, 0:total_width] = display_video

        # 5. Kural Durumlarının Hesaplanması (2.5 sn soğuma / hold süresi)
        fps_calc = 1.0 / max(0.001, time.time() - t_prev)
        t_prev = time.time()

        rule_states = {}
        for rule in RULE_DEFINITIONS:
            r_key = rule["key"]
            last_t = last_trigger_timestamps[r_key]
            dt = timestamp - last_t
            # Tetiklenme anından itibaren hold_sec (varsayılan 2.5s) boyunca AKTİF (Yeşil)
            is_active = (0.0 <= dt <= hold_sec)
            time_left = max(0.0, hold_sec - dt) if is_active else 0.0
            rule_states[r_key] = {
                "is_active": is_active,
                "time_left": time_left,
                "detail": last_trigger_details.get(r_key, ""),
            }

        # 6. Alt Şeridin Çizilmesi
        draw_bottom_panel(
            combined_frame=combined_frame,
            y_offset=height,
            bar_height=bar_height,
            video_name=video_path.name,
            frame_idx=frame_idx,
            total_frames=total_frames,
            fps=fps_calc,
            active_tracks=len(obs.get("tracks", [])),
            rule_states=rule_states,
            current_timestamp=timestamp,
        )

        # 7. Video Kaydı ve Görüntüleme
        if out_writer:
            out_writer.write(combined_frame)

        cv2.imshow(window_name, combined_frame)

        # Klavye Kontrolleri
        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key in (ord('q'), 27):  # 'q' veya ESC
            break
        elif key == ord(' '):      # SPACE
            paused = not paused
        elif key in (ord('b'), ord('B')):
            show_boxes = not show_boxes
        elif key in (ord('d'), ord('D')):
            show_graph = not show_graph
        elif key in (ord('t'), ord('T')):
            show_trails = not show_trails

    cap.release()
    if out_writer:
        out_writer.release()
        print(f"💾 Video başarıyla kaydedildi: {out_path}")
    cv2.destroyAllWindows()
    print(f"\n✅ Video Oynatımı Tamamlandı: {video_path.name}")


# =====================================================================
# SÜRÜKLE-BIRAK VE DOSYA SEÇİCİ ARAYÜZÜ (LAUNCHER GUI)
# =====================================================================

def launch_video_gui(
    config_path: str = "config.yaml",
    save_output: bool = True,
    output_dir: Path | str | None = None,
    stride: int = 1,
    conf: float = 0.25,
    proximity_threshold: float | None = None,
    hold_sec: float = ALERT_HOLD_DURATION_SECONDS,
) -> None:
    """Videoyu sürükle-bırak veya tek tıkla seçip başlatmak için sade arayüz."""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    try:
        import windnd
        has_windnd = True
    except ImportError:
        has_windnd = False

    root = tk.Tk()
    root.title("DALGA AI — İSG Video Seçici")
    root.geometry("540x460")
    root.configure(bg="#111318")
    root.resizable(False, False)

    # Pencereyi ekranın ortasına konumlandır
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = max(0, (sw - 540) // 2)
    y = max(0, (sh - 460) // 2)
    root.geometry(f"540x460+{x}+{y}")

    def on_select_path(path_str: str):
        if not path_str:
            return
        p = Path(path_str.strip('"').strip("'"))
        if not p.exists():
            messagebox.showerror("Hata", f"Video dosyası bulunamadı:\n{p}")
            return

        # Arayüzü gizle ve analizi başlat
        root.withdraw()
        try:
            run_visualizer(
                p,
                config_path=config_path,
                save_output=save_output,
                output_dir=output_dir,
                stride=stride,
                conf=conf,
                proximity_threshold=proximity_threshold,
                hold_sec=hold_sec,
            )
        except Exception as e:
            messagebox.showerror("Hata", f"Analiz sırasında bir hata oluştu:\n{e}")
        finally:
            root.deiconify()

    def browse_file():
        filetypes = [
            ("Video Dosyaları", "*.mp4 *.avi *.mkv *.mov *.webm *.m4v"),
            ("Tüm Dosyalar", "*.*"),
        ]
        chosen = filedialog.askopenfilename(
            title="Analiz Edilecek Videoyu Seçin",
            filetypes=filetypes,
        )
        if chosen:
            on_select_path(chosen)

    def on_drop(files):
        if not files:
            return
        f = files[0]
        if isinstance(f, bytes):
            f = f.decode("utf-8", errors="replace")
        on_select_path(str(f))

    # 1. Başlık Alanı
    header_frame = tk.Frame(root, bg="#111318")
    header_frame.pack(fill="x", padx=24, pady=(20, 12))

    lbl_title = tk.Label(
        header_frame,
        text="DALGA AI | İSG VİDEO ANALİZ BAŞLATICI",
        font=("Segoe UI", 13, "bold"),
        fg="#00e5ff",
        bg="#111318",
    )
    lbl_title.pack(anchor="w")

    lbl_sub = tk.Label(
        header_frame,
        text="Herhangi bir videoyu sürükleyip bırakın veya listeden seçin",
        font=("Segoe UI", 9),
        fg="#8e9aaf",
        bg="#111318",
    )
    lbl_sub.pack(anchor="w", pady=(2, 0))

    # 2. Sürükle - Bırak Alanı (Drag & Drop Zone)
    drop_frame = tk.Frame(
        root,
        bg="#1c202a",
        highlightbackground="#00e5ff",
        highlightthickness=2,
        cursor="hand2",
    )
    drop_frame.pack(fill="x", padx=24, pady=10, ipady=24)
    drop_frame.bind("<Button-1>", lambda e: browse_file())

    lbl_drop_icon = tk.Label(
        drop_frame,
        text="🎬 ⬇️",
        font=("Segoe UI", 26),
        fg="#00e5ff",
        bg="#1c202a",
    )
    lbl_drop_icon.pack(pady=(0, 4))
    lbl_drop_icon.bind("<Button-1>", lambda e: browse_file())

    lbl_drop_text = tk.Label(
        drop_frame,
        text="Videoyu Buraya Sürükleyip Bırakın",
        font=("Segoe UI", 12, "bold"),
        fg="#ffffff",
        bg="#1c202a",
    )
    lbl_drop_text.pack()
    lbl_drop_text.bind("<Button-1>", lambda e: browse_file())

    lbl_drop_hint = tk.Label(
        drop_frame,
        text="veya bilgisayardan dosya seçmek için tıklayın (.mp4, .avi, .mkv)",
        font=("Segoe UI", 9),
        fg="#94a3b8",
        bg="#1c202a",
    )
    lbl_drop_hint.pack(pady=(4, 0))
    lbl_drop_hint.bind("<Button-1>", lambda e: browse_file())

    # Windows Drag & Drop Kancası
    if has_windnd:
        try:
            windnd.hook_dropfiles(drop_frame, func=on_drop)
            windnd.hook_dropfiles(root, func=on_drop)
        except Exception:
            pass

    # 3. Hazır Test Videoları (Klasördeki .mp4'ler)
    test_videos_dir = Path(__file__).resolve().parent
    mp4_files = sorted(list(test_videos_dir.glob("*.mp4")))

    if mp4_files:
        quick_frame = tk.Frame(root, bg="#111318")
        quick_frame.pack(fill="x", padx=24, pady=(10, 14))

        lbl_quick = tk.Label(
            quick_frame,
            text="⚡ Klasördeki Test Videoları (Tek Tıkla Başlat):",
            font=("Segoe UI", 9, "bold"),
            fg="#cbd5e1",
            bg="#111318",
        )
        lbl_quick.pack(anchor="w", pady=(0, 6))

        btn_container = tk.Frame(quick_frame, bg="#111318")
        btn_container.pack(fill="x")

        for idx, vid_file in enumerate(mp4_files[:6]):
            row = idx // 3
            col = idx % 3
            btn = tk.Button(
                btn_container,
                text=vid_file.name,
                font=("Segoe UI", 9),
                fg="#f1f5f9",
                bg="#222838",
                activebackground="#00b4d8",
                activeforeground="#ffffff",
                relief="flat",
                bd=0,
                padx=8,
                pady=6,
                cursor="hand2",
                command=lambda vf=vid_file: on_select_path(str(vf)),
            )
            btn.grid(row=row, column=col, padx=4, pady=3, sticky="ew")
            btn_container.grid_columnconfigure(col, weight=1)

    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="TEKNOFEST 2026 Canlı Video İSG Kural Denetim Görselleştirici")
    parser.add_argument("--video", type=str, default=None,
                        help="Analiz edilecek video yolu/adı (Belirtilmezse seçim arayüzü açılır)")
    parser.add_argument("--config", type=str, default="config.yaml", help="config.yaml yolu")
    parser.add_argument("--stride", type=int, default=1, help="Kare atlama adımı (1 = her kare, 2 = 2 karede bir)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO güven eşiği (Varsayılan: 0.25)")
    parser.add_argument("--proximity", type=float, default=None,
                        help="Yakınlık eşiğini piksel cinsinden override eder")
    parser.add_argument("--hold-sec", type=float, default=ALERT_HOLD_DURATION_SECONDS,
                        help=f"Tetiklenen kuralın yeşil kalacağı süre (Varsayılan: {ALERT_HOLD_DURATION_SECONDS} sn)")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True,
                        help="Görselleştirilmiş videoyu İndirilenler klasörüne kaydet (Varsayılan: True)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Videonun kaydedileceği özel dizin (Varsayılan: İndirilenler / Downloads)")
    args = parser.parse_args()

    # Eğer video parametresi verilmemişse sade seçim arayüzünü aç
    if args.video is None:
        launch_video_gui(
            config_path=args.config,
            save_output=args.save,
            output_dir=args.output_dir,
            stride=args.stride,
            conf=args.conf,
            proximity_threshold=args.proximity,
            hold_sec=args.hold_sec,
        )
        return

    video_dir = Path(__file__).resolve().parent
    video_path = video_dir / args.video if not Path(args.video).is_absolute() else Path(args.video)

    run_visualizer(
        video_path,
        config_path=args.config,
        save_output=args.save,
        output_dir=args.output_dir,
        stride=args.stride,
        conf=args.conf,
        proximity_threshold=args.proximity,
        hold_sec=args.hold_sec,
    )


if __name__ == "__main__":
    main()

