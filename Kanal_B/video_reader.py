"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DOSYA: video_reader.py                                                      ║
║  KATMAN: Veri Kaynağı — plan/01 §1.1                                        ║
║  ROL   : Ham video karesini okuyup Kanal A ve Kanal B'ye sunar.              ║
║          Kanal A ve Kanal B bu jeneratörü paylaşır; her kanal kendi          ║
║          örnekleme/işleme mantığını üstüne kurar.                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

VERİ AKIŞI İÇİNDEKİ YERİ:
  [Video Dosyası] → read_video_stream() → RawFrame → preprocessing.py

NEDEN PyAV (av), cv2.VideoCapture değil?
  - cv2.VideoCapture bazı container/codec kombinasyonlarında PTS (zaman damgası)
    okumalarında hata yapar veya tutarsız sonuç döner.
  - PyAV doğrudan decoder PTS'ine eriştiği için VFR (değişken kare hızlı)
    videolarda bile doğru timestamp_sec üretir.
  - thread_type="AUTO" ile çok çekirdekli decode desteği sağlanabilir.

SINIFLAR VE FONKSİYONLAR:
  RawFrame        : Tek kare = piksel verisi (RGB numpy) + meta bilgisi
  VideoReader     : Yeniden kullanılabilir sınıf (context manager), kare sayısı tahmini yapar
  read_video_stream() : preprocessing.py'nin import ettiği basit jeneratör arayüzü
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# RawFrame — her iki kanal (A ve B) için ortak ham kare veri yapısı
# ---------------------------------------------------------------------------

@dataclass
class RawFrame:
    """Tek bir video karesinin ham verisi ve konum meta bilgisi.

    preprocessing.py bu dataclass'ı alır; kendi _Candidate iç yapısına sarar.
    Kanal A da aynı RawFrame'i kendi örnekleme mantığında kullanır.

    frame_index  : 0'dan başlayan karenin sıra numarası (stride filtresi için)
    timestamp_sec: PTS'ten hesaplanan gerçek zaman (MIN_GAP_SEC kararı için)
    rgb          : (H, W, 3) uint8 numpy dizisi, RGB sırasında (BGR değil!)
    """
    frame_index: int           # videonun 0-bazlı kare indeksi
    timestamp_sec: float       # videonun başından itibaren saniye
    rgb: NDArray[np.uint8]     # (H, W, 3) RGB24 — cv2 fonksiyonları BGR bekler, dikkat!


# ---------------------------------------------------------------------------
# VideoReader — gelişmiş kullanım için yeniden kullanılabilir sınıf
# ---------------------------------------------------------------------------

class VideoReader:
    """PyAV tabanlı güvenli video okuyucu sınıfı.

    Doğrudan kullanım (gelişmiş ihtiyaçlar için):
        with VideoReader("video.mp4") as vr:
            print(vr.total_frames, vr.fps)
            for frame_rgb in vr.iter_frames(max_frames=100):
                ...  # frame_rgb: (H, W, 3) numpy uint8

    Basit kullanım için read_video_stream() jeneratörünü tercih et.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video bulunamadı: {self.path}")

        # container açıkta kalır; close() veya __exit__ ile kapatılmalı
        self.container = av.open(str(self.path))
        self.stream = self.container.streams.video[0]
        self._total_frames: int = self._estimate_total_frames()
        # average_rate yoksa (bazı VFR videolar) güvenli default olarak 25 fps kullan
        self._fps: float = float(self.stream.average_rate) if self.stream.average_rate else 25.0

    # ------------------------------------------------------------------
    # Kare sayısı tahmini — üç kademeli güvenli yöntem
    # ------------------------------------------------------------------

    def _estimate_total_frames(self) -> int:
        """Container'dan kare sayısını üç farklı yöntemle tahmin eder.

        1. Yöntem: stream.frames (en hızlı, ama bazı container'larda eksik)
        2. Yöntem: duration * fps (meta veri var ama frames yoksa)
        3. Yöntem: decode ederek say (en yavaş, son çare)
        """
        # 1. Yöntem: container meta verisi
        total = self.stream.frames
        if total and total > 0:
            return total

        # 2. Yöntem: duration * fps hesabı
        duration = self.stream.duration
        time_base = self.stream.time_base
        avg_rate = self.stream.average_rate
        if duration and time_base and avg_rate:
            return int(float(duration * time_base) * float(avg_rate))

        # 3. Yöntem: kareleri tek tek say, sonra stream'i başa sar
        total = sum(1 for _ in self.container.decode(video=0))
        self.container.seek(0)
        return total or 1

    @property
    def total_frames(self) -> int:
        """Toplam tahmini kare sayısı."""
        return self._total_frames

    @property
    def fps(self) -> float:
        """Saniyedeki kare sayısı (ortalama)."""
        return self._fps

    def duration_seconds(self) -> float:
        """Video süresini saniye cinsinden döner."""
        return self._total_frames / self._fps if self._fps else 0.0

    def iter_frames(self, max_frames: int | None = None) -> Iterator[NDArray[np.uint8]]:
        """RGB24 numpy dizisi olarak kareleri döner (RawFrame meta bilgisi olmadan).

        Sadece piksel verisi gerekiyorsa kullan.
        timestamp_sec ve frame_index de gerekiyorsa read_video_stream() kullan.
        """
        self.container.seek(0)
        count = 0
        for frame in self.container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")
            count += 1
            if max_frames is not None and count >= max_frames:
                break

    def close(self) -> None:
        """Container'ı kapat, belleği serbest bırak."""
        self.container.close()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# read_video_stream — preprocessing.py'nin kullandığı birincil arayüz
# ---------------------------------------------------------------------------

def read_video_stream(video_path: str) -> Iterator[RawFrame]:
    """Videoyu baştan sona akış halinde okur; her kare için RawFrame döner.

    preprocessing.py bu jeneratörü şöyle kullanır:
        for frame in read_video_stream(video_path):
            if frame.frame_index % CANDIDATE_STRIDE != 0:
                continue           # her 5. kareyi işle (CANDIDATE_STRIDE=5)
            ...

    NEDEN BU ARAYÜZ?
      - Videoyu belleğe tamamen yüklemez (streaming) → büyük videolar için verimli
      - frame_index: stride filtresi için (her N. kare)
      - timestamp_sec: MIN_GAP_SEC kontrolü için (seçilen kareler arası min süre)
      - rgb: cv2 ve skimage işlemleri için direkt kullanılabilir numpy array

    PTS → timestamp_sec dönüşümü:
      pts_in_seconds = frame.pts * stream.time_base
      pts yoksa (bozuk container) → frame_index / fps ile tahmin edilir.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video bulunamadı: {path}")

    with av.open(str(path)) as container:
        stream = container.streams.video[0]

        # time_base: PyAV'ın PTS birimini saniyeye çeviren kesir (ör. 1/90000)
        time_base = float(stream.time_base) if stream.time_base else 1.0

        frame_index = 0
        for av_frame in container.decode(video=0):
            # PTS mevcut değilse (bazı bozuk/eksik container'lar) indeks/fps kullan
            if av_frame.pts is not None:
                timestamp_sec = float(av_frame.pts) * time_base
            else:
                fps = float(stream.average_rate) if stream.average_rate else 25.0
                timestamp_sec = frame_index / fps

            # RGB24 formatında numpy array — preprocessing.py cv2 ile işler
            rgb = av_frame.to_ndarray(format="rgb24")

            yield RawFrame(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                rgb=rgb,
            )
            frame_index += 1
