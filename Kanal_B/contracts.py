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
class RiskEvent:
    """VLM'in kendi gözüyle tespit ettiği tek bir risk durumu.

    Düz bir metin listesi (eski ``risk_flags_tr``) Karar Ajanı'na riskin ne
    kadar kritik olduğunu ve kaç kanıtla desteklendiğini söylemiyordu. Bu yapı
    her riski şiddet + kanıt bilgisiyle taşır, böylece karar ajanı kanıt
    ağırlıklandırması yapabilir.

    İki kanıt biçimi desteklenir:

    * **Grid modu** — kareler tek mozaikte gönderildiğinde riskin görüldüğü
      hücre indeksleri ``supporting_frame_positions`` alanına yazılır.
    * **Video modu** — video doğrudan gönderildiğinde riskin gerçekleştiği an
      ``timestamp_sec`` alanına, tam videonun başından itibaren **mutlak**
      saniye olarak yazılır (segment ofseti uygulanmış hâlde).
    """
    description_tr: str
    severity: Literal["low", "medium", "high", "critical"]
    confidence: float                              # 0-1, riske özgü güven
    supporting_frame_count: int = 1                # kaç karede/hücrede görüldü
    supporting_frame_positions: list[int] = field(default_factory=list)
    timestamp_sec: float | None = None             # video modu: mutlak saniye


#: :attr:`RiskEvent.severity` değerlerinin sayısal karşılığı. Segmentleri
#: kritiklik sırasına dizmek ve en kritik segmenti bulmak için kullanılır.
SEVERITY_RANK: dict[str, int] = {"low": 1, "medium": 2, "high": 3, "critical": 4}


@dataclass
class VLMInterpretation:
    """S8 sözleşmesi — Karar Ajanı'nın 3. GİRDİSİ.

    backend.py bu paketi üretir; Karar Ajanı bunu alır ve Kanal A'nın
    çıktısıyla (S2, S3) birleştirerek nihai kararı verir.

    packet_id → S1b ile eşleşerek tam izlenebilirlik sağlar.
    scene_summary_tr → olay motorundan BAĞIMSIZ Türkçe sahne yorumu (kritik!)
    raw_model_output → debug/audit için saklanan ham model çıktısı.

    Alan sırası notu: varsayılanı olan alanlar sonda toplanmıştır. Tüm
    üreticiler (backend.py) anahtar kelimeli çağrı kullandığı için bu sıralama
    çağıranları etkilemez.
    """
    packet_id: str                 # S1b VLMFramePacket.packet_id ile eşleşir
    video_id: str                  # hangi videodan üretildiği
    model_name: str                # kullanılan VLM modelinin adı
    # "server" = OpenAI-uyumlu harici servis (TEKNOFEST EVREN dahil)
    model_backend: Literal["vllm", "llama.cpp", "transformers", "server"]
    scene_summary_tr: str          # 1-3 cümle Türkçe sahne özeti
    detected_entities: list[DetectedEntity]   # tespit edilen varlıklar
    detected_actions_tr: list[str]            # gözlemlenen eylemler (Türkçe)
    confidence_overall: float      # 0-1 modelin öz güven değerlendirmesi
    inference: InferenceMetrics    # performans metrikleri
    raw_model_output: str          # ham LLM çıktısı (debug amaçlı)

    # --- varsayılanı olan alanlar ---
    risk_events: list[RiskEvent] = field(default_factory=list)
    # Geriye dönük uyumluluk: eski tüketiciler (src/reasoning/decision_agent.py,
    # backend/decision/main.py) düz metin listesi okur. Boş bırakılırsa
    # __post_init__ bunu risk_events'ten türetir; böylece her tüketici çalışır.
    risk_flags_tr: list[str] = field(default_factory=list)
    # Modelin öz güveninden BAĞIMSIZ, dışsal sinyallere dayanan güven skoru.
    computed_confidence: float = 0.0
    # <think>...</think> içeriği (varsa); ayrıştırıcı izi sildiği için genelde boş.
    reasoning_trace: str = ""
    # Bu yorumun tam video içindeki mutlak zaman aralığı. Segment bazlı
    # iterasyonda her segment kendi ofsetini taşır; tek parça analizde 0-süre.
    segment_start_sec: float = 0.0
    segment_end_sec: float = 0.0
    created_at: str = field(default_factory=_now_iso)

    def __post_init__(self) -> None:
        """``risk_flags_tr`` boşsa ``risk_events``ten türetir.

        Yeni üreticiler yalnızca ``risk_events`` doldurur; eski tüketiciler ise
        ``risk_flags_tr`` okur. Türetmeyi burada bir kez yapmak, her tüketicide
        aynı dönüşümü tekrarlamayı önler.
        """
        if not self.risk_flags_tr and self.risk_events:
            self.risk_flags_tr = [
                ev.description_tr for ev in self.risk_events if ev.description_tr
            ]

    @property
    def max_severity(self) -> str:
        """Bu yorumdaki en yüksek risk şiddeti; risk yoksa ``"low"``."""
        if not self.risk_events:
            return "low"
        return max(self.risk_events, key=lambda e: SEVERITY_RANK.get(e.severity, 0)).severity

    def to_dict(self) -> dict:
        """JSON serializasyonu — Karar Ajanı'na bu dict iletilir."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Segment bazlı (iterasyonlu) analiz sözleşmeleri
# ---------------------------------------------------------------------------

def format_mmss(seconds: float) -> str:
    """Saniyeyi ``MM:SS`` biçimine çevirir.

    Zaman damgaları hem prompt'a hem nihai çıktıya bu biçimde girer; karar
    ajanının şeması da ``MM:SS`` beklediği için dönüşüm tek yerde tutulur.
    """
    total = max(0, int(round(seconds)))
    return f"{total // 60:02d}:{total % 60:02d}"


@dataclass
class SegmentResult:
    """Tek bir video segmentinin analiz sonucu.

    Uzun videolar 60 saniyelik parçalara bölünüp sırayla incelenir; her parça
    için bir :class:`SegmentResult` üretilir. ``start_sec``/``end_sec`` tam
    videoya göre **mutlak** zamanı verir, böylece segmentler birleştirildiğinde
    olayların gerçek anı korunur.
    """
    segment_index: int
    start_sec: float
    end_sec: float
    interpretation: VLMInterpretation

    @property
    def duration_sec(self) -> float:
        """Segmentin süresi (saniye)."""
        return max(0.0, self.end_sec - self.start_sec)

    @property
    def max_severity(self) -> str:
        """Segmentteki en yüksek risk şiddeti."""
        return self.interpretation.max_severity

    def to_dict(self) -> dict:
        """JSON serializasyonu."""
        return {
            "segment_index": self.segment_index,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "time_range": f"{format_mmss(self.start_sec)}-{format_mmss(self.end_sec)}",
            "max_severity": self.max_severity,
            "interpretation": self.interpretation.to_dict(),
        }


@dataclass
class BatchResult:
    """Segment segment incelenmiş bir videonun toplu sonucu.

    Karar Ajanı'na tek bir S8 yorumu yerine bu toplu sonuç iletilebilir;
    :meth:`to_interpretation_dict` ile tek yoruma indirgenmiş hâli de alınabilir
    (mevcut tüketiciler tek yorum beklediği için).
    """
    video_id: str
    segments: list[SegmentResult] = field(default_factory=list)
    #: İşlenemeyen segmentler. Sessiz veri kaybı yerine burada raporlanır;
    #: karar ajanı analizin kısmi olduğunu bilerek güven ayarlar.
    failed_segments: list[dict] = field(default_factory=list)
    #: Segmentler arası taşınan metin hafızasının son hâli.
    memory_context: str = ""
    created_at: str = field(default_factory=_now_iso)

    @property
    def total_segments(self) -> int:
        """Başarıyla işlenen segment sayısı."""
        return len(self.segments)

    @property
    def total_risk_events(self) -> int:
        """Tüm segmentlerdeki risk olaylarının toplamı."""
        return sum(len(s.interpretation.risk_events) for s in self.segments)

    @property
    def most_critical_segment(self) -> SegmentResult | None:
        """En yüksek şiddetli riski içeren segment; risk yoksa ``None``.

        Eşitlikte, riske özgü güveni yüksek olan segment seçilir.
        """
        scored = [s for s in self.segments if s.interpretation.risk_events]
        if not scored:
            return None
        return max(
            scored,
            key=lambda s: (
                SEVERITY_RANK.get(s.max_severity, 0),
                max((e.confidence for e in s.interpretation.risk_events), default=0.0),
            ),
        )

    @property
    def all_risk_events(self) -> list[RiskEvent]:
        """Tüm segmentlerin risk olayları, zaman sırasında tek listede."""
        events: list[RiskEvent] = []
        for seg in sorted(self.segments, key=lambda s: s.start_sec):
            events.extend(seg.interpretation.risk_events)
        return events

    def to_dict(self) -> dict:
        """JSON serializasyonu."""
        critical = self.most_critical_segment
        return {
            "video_id": self.video_id,
            "total_segments": self.total_segments,
            "total_risk_events": self.total_risk_events,
            "most_critical_segment_index": critical.segment_index if critical else None,
            "memory_context": self.memory_context,
            "segments": [s.to_dict() for s in self.segments],
            "failed_segments": self.failed_segments,
            "created_at": self.created_at,
        }

    def to_interpretation_dict(self) -> dict:
        """Toplu sonucu **tek** S8 yorum sözlüğüne indirger.

        Mevcut tüketiciler (:mod:`src.reasoning.decision_agent`,
        ``backend/decision/main.py``) tek bir yorum sözlüğü bekler. Bu metot
        segmentleri birleştirir: özetler zaman aralıklarıyla etiketlenip
        birleştirilir, risk olayları mutlak zamanlarıyla toplanır, güven
        skorları ortalanır.

        Returns:
            :meth:`VLMInterpretation.to_dict` ile aynı anahtarlara sahip sözlük.
        """
        if not self.segments:
            return {
                "scene_summary_tr": "Video analiz edilemedi.",
                "detected_entities": [],
                "detected_actions_tr": [],
                "risk_events": [],
                "risk_flags_tr": [],
                "confidence_overall": 0.0,
                "computed_confidence": 0.0,
                "segment_count": 0,
                "failed_segments": self.failed_segments,
            }

        ordered = sorted(self.segments, key=lambda s: s.start_sec)

        summary_parts: list[str] = []
        actions: list[str] = []
        entities: list[dict] = []
        seen_entities: set[str] = set()
        risk_events: list[dict] = []

        for seg in ordered:
            interp = seg.interpretation
            if interp.scene_summary_tr:
                label = f"[{format_mmss(seg.start_sec)}-{format_mmss(seg.end_sec)}]"
                summary_parts.append(f"{label} {interp.scene_summary_tr}")
            actions.extend(interp.detected_actions_tr)
            for ent in interp.detected_entities:
                if ent.label and ent.label.lower() not in seen_entities:
                    seen_entities.add(ent.label.lower())
                    entities.append(asdict(ent))
            for ev in interp.risk_events:
                item = asdict(ev)
                # Segment yorumu zaman damgasını üretmediyse segment başlangıcı
                # en iyi tahmindir; aksi hâlde olay zamansız kalır.
                if item.get("timestamp_sec") is None:
                    item["timestamp_sec"] = seg.start_sec
                item["timestamp"] = format_mmss(float(item["timestamp_sec"]))
                risk_events.append(item)

        risk_events.sort(key=lambda e: e.get("timestamp_sec") or 0.0)
        n = len(ordered)

        return {
            "scene_summary_tr": " ".join(summary_parts),
            "detected_entities": entities,
            "detected_actions_tr": actions,
            "risk_events": risk_events,
            "risk_flags_tr": [e["description_tr"] for e in risk_events if e.get("description_tr")],
            "confidence_overall": sum(s.interpretation.confidence_overall for s in ordered) / n,
            "computed_confidence": sum(s.interpretation.computed_confidence for s in ordered) / n,
            "segment_count": n,
            "failed_segments": self.failed_segments,
            "memory_context": self.memory_context,
        }


# ---------------------------------------------------------------------------
# Metin tabanlı anlamsal + zamansal hafıza (iterasyonlu analiz için)
# ---------------------------------------------------------------------------

@dataclass
class VideoAnalysisMemory:
    """Segmentler arasında taşınan, **metin tabanlı** anlamsal ve zamansal hafıza.

    Neden metin, neden sınırlı?
      Uzun bir video 60 saniyelik parçalara bölünüp sırayla incelenir. Sonraki
      segmentin bağlamı olmadan yorumlanması, süregelen durumları (aynı kişi,
      aynı araç, devam eden risk) görmezden gelir. Öte yandan önceki
      segmentlerin **ham çıktılarını** biriktirmek bağlam penceresini hızla
      doldurur. Bu sınıf ikisinin arasını tutar: her segmentten yalnızca
      damıtılmış metin satırları saklanır ve satır sayısı üst sınırla
      kırpılır. Böylece bağlam maliyeti segment sayısıyla doğrusal büyümek
      yerine sabit bir tavana oturur.

    İki tür hafıza ayrı tutulur:
      * **Anlamsal** (:attr:`semantic_notes`): sahnenin ve varlıkların kalıcı
        nitelikleri — "depo içi, raflı alan", "beyaz kasklı personel".
      * **Zamansal** (:attr:`temporal_events`): mutlak zaman damgalı olay
        satırları — "01:12 — personel merdivenden düştü [high]".
    """
    semantic_notes: list[str] = field(default_factory=list)
    temporal_events: list[str] = field(default_factory=list)
    #: Saklanacak azami anlamsal not sayısı (en yeniler korunur).
    max_semantic_notes: int = 12
    #: Saklanacak azami zamansal olay satırı (en yeniler korunur).
    max_temporal_events: int = 40
    #: Tek bir anlamsal notun azami karakter uzunluğu.
    max_note_chars: int = 240

    def absorb(self, interpretation: VLMInterpretation, segment_start_sec: float = 0.0) -> None:
        """Bir segment yorumunu damıtıp hafızaya ekler.

        Args:
            interpretation: Segmentin S8 yorumu.
            segment_start_sec: Segmentin tam videodaki mutlak başlangıcı;
                yorumun kendi zaman damgası yoksa olay zamanı buna dayanır.
        """
        summary = (interpretation.scene_summary_tr or "").strip()
        if summary:
            note = f"[{format_mmss(segment_start_sec)}] {summary[: self.max_note_chars]}"
            if note not in self.semantic_notes:
                self.semantic_notes.append(note)

        labels = [e.label for e in interpretation.detected_entities if e.label]
        if labels:
            uniq = sorted(set(labels))
            note = f"[{format_mmss(segment_start_sec)}] görülen varlıklar: {', '.join(uniq)}"
            if note not in self.semantic_notes:
                self.semantic_notes.append(note)

        for ev in interpretation.risk_events:
            when = ev.timestamp_sec if ev.timestamp_sec is not None else segment_start_sec
            line = (
                f"{format_mmss(float(when))} — {ev.description_tr} "
                f"[{ev.severity}, güven {ev.confidence:.2f}]"
            )
            if line not in self.temporal_events:
                self.temporal_events.append(line)

        self._trim()

    def _trim(self) -> None:
        """Hafızayı üst sınırlara kırpar (en yeni kayıtlar korunur)."""
        if len(self.semantic_notes) > self.max_semantic_notes:
            self.semantic_notes = self.semantic_notes[-self.max_semantic_notes :]
        if len(self.temporal_events) > self.max_temporal_events:
            self.temporal_events = self.temporal_events[-self.max_temporal_events :]

    def is_empty(self) -> bool:
        """Hafızada hiç kayıt yoksa ``True``."""
        return not self.semantic_notes and not self.temporal_events

    def to_prompt_context(self) -> str:
        """Hafızayı sonraki segmentin prompt'una eklenecek metne çevirir.

        Returns:
            Boş hafızada boş dize; aksi hâlde iki başlıklı özet metin.
        """
        if self.is_empty():
            return ""
        lines = ["ÖNCEKİ SEGMENTLERDEN TAŞINAN BAĞLAM (metin hafıza):"]
        if self.semantic_notes:
            lines.append("Sahne ve varlıklar:")
            lines.extend(f"  - {n}" for n in self.semantic_notes)
        if self.temporal_events:
            lines.append("Zaman damgalı olaylar:")
            lines.extend(f"  - {e}" for e in self.temporal_events)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON serializasyonu."""
        return {
            "semantic_notes": list(self.semantic_notes),
            "temporal_events": list(self.temporal_events),
        }
