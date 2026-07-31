"""
plan/13 sözleşmeleri: S1b (vlm_frame_packet) ve S8 (vlm_interpretation)

NOT: Bu şemalar diagramdan ve görev tanımından türetildi (plan/13 dosyanın
içeriğini görmedim). Eğer gerçek sözleşmede alan adları/isimlendirme farklıysa,
sadece bu dosyadaki dataclass alanlarını güncellemen yeterli — üretim mantığı
(preprocessing.py, vlm_backend.py) şemadan ayrıştırılmış durumda, sadece
to_dict() çıktısını eşlemen gerekir.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Literal
import uuid


def _now_iso() -> str:
    """UTC ISO-8601 zaman damgası üretir — tüm paketler için oluşturulma zamanı."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# S1b — Kanal B ön işleme çıktısı (preprocessing.py → backend.py)
# ---------------------------------------------------------------------------

@dataclass
class FrameQualityMetrics:
    """Bir kare için hesaplanan kalite/sahne metrikleri.
    Bu metrikler hem kare seçim kararını hem de S1b paketinin içeriğini belirler."""
    laplacian_var: float          # keskinlik / bulanıklık tespiti — yüksek = keskin
    ssim_diff: float               # bir önceki seçilen kareyle sahne farkı (0-1)
    brightness_mean: float         # 0-255, düşük ışık kararı için (CLAHE eşiği)


@dataclass
class FrameMeta:
    """Seçilen her kare için kimlik + seçilme gerekçesi + kalite bilgisi.
    selection_reason → Karar Ajanı'nın hangi kareye neden bakması gerektiğini anlar."""
    frame_index: int               # orijinal videodaki karenin sıra numarası
    timestamp_sec: float           # videonun başından itibaren saniye cinsinden konum
    grid_position: int             # grid içindeki sırası (0'dan başlar, soldan sağa)
    selection_reason: Literal[
        "scene_change",            # SSIM farkı yüksek → sahne değişimi
        "motion_peak",             # optik akış zirvesi (ileride eklenecek)
        "uniform_sample",          # SSIM farkı düşük → eşit aralıklı örnekleme
        "fallback_fill",           # yetersiz aday → zaman ekseninde eşit aralıklı doldurma
    ]
    quality: FrameQualityMetrics


@dataclass
class GridLayout:
    """Grid görseli boyut bilgisi — VLM hangi boyutta görüntü aldığını bilsin."""
    rows: int                      # satır sayısı (genellikle 2)
    cols: int                      # sütun sayısı (genellikle 4)
    cell_size: tuple[int, int]     # tek hücre (w, h) piksel — config.yaml'dan gelir


@dataclass
class EnhancementInfo:
    """CLAHE uygulaması meta verisi — Karar Ajanı düşük ışık düzeltmesi yapıldığını bilsin."""
    clahe_applied: bool            # en az bir kare karanlıksa True
    clip_limit: float              # CLAHE kontrast sınır katsayısı
    tile_grid_size: tuple[int, int]  # CLAHE mozaik ızgara boyutu


@dataclass
class VLMFramePacket:
    """S1b sözleşmesi — Kanal B ön işleme çıktısı, VLM backend'in GİRDİSİ.

    preprocessing.py bu paketi üretir; backend.py bunu alır ve VLM'e gönderir.
    packet_id, S1b ile S8 arasındaki izlenebilirlik bağlantısını kurar."""
    video_id: str                  # hangi videodan üretildiği
    source_start_sec: float        # seçilen ilk karenin zaman damgası
    source_end_sec: float          # seçilen son karenin zaman damgası
    frames: list[FrameMeta]        # seçilen her karenin meta bilgisi
    grid_layout: GridLayout        # grid boyut bilgisi
    enhancement: EnhancementInfo   # CLAHE uygulama bilgisi
    grid_image_path: str           # disk üzerindeki grid JPEG'inin tam yolu
    packet_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # UUID
    created_at: str = field(default_factory=_now_iso)  # UTC oluşturulma zamanı

    def to_dict(self) -> dict:
        """JSON serializasyonu — tüm iç içe dataclass'ları dict'e çevirir."""
        return asdict(self)


# ---------------------------------------------------------------------------
# S8 — VLM çıktısı (backend.py → Karar Ajanı)
# ---------------------------------------------------------------------------

@dataclass
class DetectedEntity:
    """VLM'in tespit ettiği bir nesne/varlık.
    label: kısa etiket (araç, kişi, yük…) — VLM kesin tür tahmini yapmamalı.
    confidence_hint: modelin kendi güven değerlendirmesi (low/medium/high)."""
    label: str
    confidence_hint: Literal["low", "medium", "high"]
    notes_tr: str = ""             # ek açıklama (isteğe bağlı, Türkçe)


@dataclass
class InferenceMetrics:
    """VLM çağrısının performans metrikleri — izleme ve optimizasyon için."""
    latency_ms: float              # toplam inference süresi (milisaniye)
    tokens_generated: int          # üretilen token sayısı


@dataclass
class VLMInterpretation:
    """S8 sözleşmesi — Karar Ajanı'nın 3. GİRDİSİ. YAPI SABİT, DIŞINA ÇIKMA.

    backend.py bu paketi üretir; Karar Ajanı (plan/05) bunu alır ve
    Kanal A'nın çıktısıyla (S2, S3) birleştirerek nihai kararı verir.

    packet_id → S1b ile eşleşerek tam izlenebilirlik sağlar.
    scene_summary_tr → olay motorundan BAĞIMSIZ Türkçe sahne yorumu (kritik!)
    raw_model_output → debug/audit için saklanan ham model çıktısı."""
    packet_id: str                 # S1b VLMFramePacket.packet_id ile eşleşir
    video_id: str                  # hangi videodan üretildiği
    model_name: str                # kullanılan VLM modelinin adı
    model_backend: Literal["vllm", "llama.cpp", "transformers"]  # çalıştırıcı
    scene_summary_tr: str          # 1-3 cümle Türkçe sahne özeti
    detected_entities: list[DetectedEntity]   # tespit edilen varlıklar
    detected_actions_tr: list[str]            # gözlemlenen eylemler (Türkçe)
    risk_flags_tr: list[str]                  # risk bayrakları (Türkçe, yoksa [])
    confidence_overall: float      # 0-1 modelin öz güven değerlendirmesi
    inference: InferenceMetrics    # performans metrikleri
    raw_model_output: str          # ham LLM çıktısı (debug amaçlı)
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        """JSON serializasyonu — Karar Ajanı'na bu dict iletilir."""
        return asdict(self)