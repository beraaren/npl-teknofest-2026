"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DOSYA: preprocessing.py                                                     ║
║  KATMAN: Kanal B Ön İşleme — plan/01 §1.3                                   ║
║  ROL   : Ham videoyu → VLM'e gönderilebilir kare paketine (S1b) dönüştürür. ║
╚══════════════════════════════════════════════════════════════════════════════╝

VERİ AKIŞI İÇİNDEKİ YERİ:
  video_reader.py → [RawFrame akışı]
    → _collect_candidates()   : Her 5. kareyi tara, metrik hesapla
    → _select_frames()        : En bilgilendirici 8 kareyi seç
    → _apply_clahe_if_needed(): Karanlık kareleri aydınlat
    → _build_grid()           : 8 kareyi 2×4 grid'e diz
    → build_vlm_frame_packet(): Hepsini S1b paketine sar → backend.py'ye gönder

NEDEN GRİD?
  VLM'e 8 ayrı görüntü yerine tek bir grid görseli gönderilir:
  - Tek HTTP çağrısı → daha hızlı
  - Model tüm zaman dilimini bir arada görür → bağlamsal yorum daha iyi
  - Mevcut grid boyutu: 4 sütun × 2 satır = 8 hücre, her hücre config'den gelir

AKILLI ÖRNEKLEME ALGORİTMASI (3 aşamalı):
  Aşama 1 — Bulanık filtre: Laplacian varyansı < eşik olan kareler elenir
  Aşama 2 — Sahne değişimi greedy seçim: SSIM farkı yüksek, zaman boşluklu kareler
  Aşama 3 — Fallback doldurma: Hâlâ 8'e ulaşılamadıysa eşit aralıklı ekle

TÜM PARAMETRELERİ config.yaml'daki [preprocessing:] bloğu belirler (S4 değişikliği).
config.yaml yoksa parantez içindeki default değerler devreye girer.
"""
from __future__ import annotations
import os
import uuid
import yaml                                          # config.yaml okuma için
from dataclasses import dataclass

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# video_reader.py'den ham kare akışı arayüzü
from video_reader import RawFrame, read_video_stream

# contracts.py'den S1b sözleşme veri yapıları
from contracts import (
    VLMFramePacket, FrameMeta, FrameQualityMetrics,
    GridLayout, EnhancementInfo,
)


# ---------------------------------------------------------------------------
# Config yükleme — tüm sabitler buradan gelir (S4 değişikliği)
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    """Proje kökündeki config.yaml'ın [preprocessing:] bloğunu yükler.

    Neden modül seviyesinde bir kez yükleniyor?
      - Her çağrıda disk okumaktan kaçınmak için
      - Değiştirildiğinde sadece config.yaml güncellenir, kod değişmez

    Dosya yoksa boş dict döner → tüm sabitler kendi default değerlerini kullanır.
    """
    cfg_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    )
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f).get("preprocessing", {})


_CFG = _load_cfg()  # modül yüklendiğinde bir kez çalışır

# ---------------------------------------------------------------------------
# Ayarlanabilir sabitler — config.yaml'dan okunur, yoksa default değer
# ---------------------------------------------------------------------------

# Kaç kare seçileceği — aynı zamanda grid hücre sayısını da belirler
TARGET_FRAME_COUNT   = _CFG.get("target_frame_count", 8)

# Her kaç karede bir aday metriği hesaplanır (5 → her 5. kare)
# Daha küçük = daha hassas ama daha yavaş; büyük videolarda artır
CANDIDATE_STRIDE     = 5  # config.yaml'da tanımlı değil, sabit kalır

# Laplacian varyansı bu değerin altındaki kareler "bulanık" sayılır ve elenir
# config.yaml: min_laplacian_variance: 80.0 — test ederek idealini bul
MIN_SHARPNESS        = _CFG.get("min_laplacian_variance", 40.0)

# Seçilen iki kare arasındaki minimum zaman farkı (saniye)
# Bu sayede tüm kareler videonun tek bir anından gelmez, zamansal çeşitlilik sağlanır
MIN_GAP_SEC          = 1.0  # config.yaml'da tanımlı değil, sabit kalır

# Ortalama parlaklık bu değerin altındaysa CLAHE uygulanır (0=siyah, 255=beyaz)
LOW_LIGHT_BRIGHTNESS = 90.0  # config.yaml'da tanımlı değil, sabit kalır

# CLAHE parametreleri — kontrast sınır katsayısı ve mozaik ızgara boyutu
CLAHE_CLIP_LIMIT     = _CFG.get("clahe_clip_limit", 2.5)
CLAHE_TILE_GRID      = tuple(_CFG.get("clahe_grid_size", [8, 8]))

# Tek grid hücresi boyutu (w, h) — config.yaml'dan gelir
# NOT: frame_height=216 kullanılırsa hücre dikdörtgen olur (384×216), 384×384 değil
_W                   = _CFG.get("frame_width", 384)
_H                   = _CFG.get("frame_height", 384)
CELL_SIZE            = (_W, _H)

# Grid düzeni — TARGET_FRAME_COUNT / GRID_COLS satır sayısını belirler
GRID_COLS            = _CFG.get("grid_columns", 4)
GRID_ROWS            = TARGET_FRAME_COUNT // GRID_COLS  # 8 // 4 = 2


# ---------------------------------------------------------------------------
# İç veri yapısı — yalnızca bu modül içinde kullanılır
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    """Kare seçim sürecindeki bir aday kare.

    _collect_candidates() tarafından üretilir, _select_frames() tarafından tüketilir.
    RawFrame'i sarmalar + hesaplanan 3 metriği tutar.
    """
    frame: RawFrame        # ham kare (frame_index, timestamp_sec, rgb)
    laplacian_var: float   # keskinlik metriği — bulanık eşik karşılaştırması için
    ssim_diff: float       # bir önceki adayla sahne farkı — değişim tespiti için
    brightness: float      # ortalama parlaklık — CLAHE kararı için


# ---------------------------------------------------------------------------
# Yardımcı hesaplama fonksiyonları
# ---------------------------------------------------------------------------

def _laplacian_var(gray: np.ndarray) -> float:
    """Gri tonlamalı kareden Laplacian varyansı hesaplar.

    Yüksek değer = keskin kare (net kenarlar var)
    Düşük değer  = bulanık kare (odak dışı ya da hareket bulanıklığı)
    MIN_SHARPNESS eşiğiyle karşılaştırılarak bulanık kareler elenir.
    """
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _to_gray_small(rgb: np.ndarray, size=(160, 90)) -> np.ndarray:
    """SSIM hesabı için küçültülmüş gri tonlamalı kare üretir.

    Neden küçültüyoruz?
      SSIM hesabı büyük görüntülerde yavaş → küçük çözünürlükte ≈ aynı sonuç,
      çok daha hızlı hesap. 160×90 px SSIM için yeterli.
    """
    small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)


# ---------------------------------------------------------------------------
# Aşama 1 — Aday havuzu oluşturma
# ---------------------------------------------------------------------------

def _collect_candidates(video_path: str) -> list[_Candidate]:
    """Videoyu bir kez tarar, her CANDIDATE_STRIDE karede bir metrik hesaplar.

    NEDEN ADAY HAVUZU?
      Önce tüm video taranır, aday metrikler hesaplanır.
      Sonra _select_frames() bu havuzdan en iyi 8'i seçer.
      Bu yaklaşım videoyu sadece bir kez okur (bellek dostu).

    Belleğe alınan şey: sadece adayların metrik değerleri + ham kare referansları
    (tüm video karelerinin piksel verisi değil).

    SSIM farkı: Her CANDIDATE_STRIDE'th kare bir öncekiyle karşılaştırılır.
      ssim_diff = 1.0 - SSIM(önceki, şimdiki)
      Yüksek ssim_diff → sahne değişimi olmuş → bu kare bilgilendirici
    """
    candidates: list[_Candidate] = []
    prev_gray = None  # SSIM için önceki küçük gri kare

    for frame in read_video_stream(video_path):
        # Her 5. kareyi al; aradakileri atla (hız optimizasyonu)
        if frame.frame_index % CANDIDATE_STRIDE != 0:
            continue

        # Tam çözünürlükte Laplacian (keskinlik), küçük çözünürlükte SSIM
        gray_full  = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        gray_small = _to_gray_small(frame.rgb)

        lap        = _laplacian_var(gray_full)
        brightness = float(gray_full.mean())

        # İlk kare için karşılaştırılacak önceki kare yok → diff=1.0 (en yüksek)
        diff = 1.0 if prev_gray is None else float(1.0 - ssim(prev_gray, gray_small))

        candidates.append(_Candidate(frame, lap, diff, brightness))
        prev_gray = gray_small

    return candidates


# ---------------------------------------------------------------------------
# Aşama 2 ve 3 — Akıllı kare seçimi + fallback doldurma
# ---------------------------------------------------------------------------

def _select_frames(
    candidates: list[_Candidate],
) -> tuple[list[_Candidate], set[int]]:
    """Aday havuzundan en bilgilendirici TARGET_FRAME_COUNT kareyi seçer.

    DÖNER: (seçilen_liste, fallback_fill_id_seti)
      fallback_ids → 3. aşamada eklenen karelerin Python id()'leri
      Bu set, build_vlm_frame_packet()'te selection_reason'ı doğru atamak için kullanılır.

    SEÇIM MANTIĞI (3 aşamalı):

    Aşama 1 — Bulanık filtre:
      Laplacian varyansı < MIN_SHARPNESS olan kareler elenir.
      Tüm kareler bulanıksa eleme yapılmaz (hiç kare kalmamasın diye).

    Aşama 2 — Greedy sahne değişimi seçimi:
      Adaylar SSIM farkına göre büyükten küçüğe sıralanır.
      Her kare için: seçilmiş tüm karelerle zaman farkı >= MIN_GAP_SEC mi?
        Evet → seç (zamansal çeşitlilik korunur)
        Hayır → atla (aynı anı iki kez alma)

    Aşama 3 — Fallback doldurma:
      Hâlâ TARGET_FRAME_COUNT'a ulaşılamadıysa, zaman ekseninde eşit aralıklı
      adaylar eklenir. Bunlar selection_reason="fallback_fill" alır.
    """
    # Bulanık kareleri ele; tümü bulanıksa hiç eleme yapma
    sharp  = [c for c in candidates if c.laplacian_var >= MIN_SHARPNESS] or candidates
    ranked = sorted(sharp, key=lambda c: c.ssim_diff, reverse=True)

    selected: list[_Candidate] = []
    for c in ranked:
        if len(selected) >= TARGET_FRAME_COUNT:
            break
        # Zaman boşluğu kontrolü: seçilmiş tüm karelerden MIN_GAP_SEC uzakta mı?
        if all(abs(c.frame.timestamp_sec - s.frame.timestamp_sec) >= MIN_GAP_SEC
               for s in selected):
            selected.append(c)

    fallback_ids: set[int] = set()  # Aşama 3'te eklenen karelerin id'leri

    if len(selected) < TARGET_FRAME_COUNT:
        remaining = TARGET_FRAME_COUNT - len(selected)
        # Zaman sırasına diz, eşit aralıklarla örnekle
        pool = sorted(candidates, key=lambda c: c.frame.timestamp_sec)
        selected_indices = {id(s) for s in selected}
        if pool:
            step = max(1, len(pool) // max(remaining, 1))
            for i in range(0, len(pool), step):
                if len(selected) >= TARGET_FRAME_COUNT:
                    break
                cand = pool[i]
                if id(cand) not in selected_indices:
                    selected.append(cand)
                    selected_indices.add(id(cand))
                    fallback_ids.add(id(cand))  # bu kare fallback ile geldi

    # Son sıralama: grid soldan-sağa zaman sırasında olsun
    selected.sort(key=lambda c: c.frame.timestamp_sec)
    return selected[:TARGET_FRAME_COUNT], fallback_ids


# ---------------------------------------------------------------------------
# Aşama 4 — Düşük ışık iyileştirme
# ---------------------------------------------------------------------------

def _apply_clahe_if_needed(rgb: np.ndarray, brightness: float) -> tuple[np.ndarray, bool]:
    """Sadece karanlık karelerde CLAHE (Kontrast Sınırlı Adaptif Histogram Eşitleme) uygular.

    NEDEN KOŞULLU?
      Parlak karelere CLAHE uygulamak görüntüyü bozar (aşırı kontrast).
      Yalnızca brightness < LOW_LIGHT_BRIGHTNESS olan kareler işlenir.

    CLAHE NEDİR?
      LAB renk uzayının L (parlaklık) kanalına histogram eşitleme uygular.
      A ve B kanalları (renk) değişmez → renk bozulması olmaz.
      clip_limit: lokal kontrast artışını sınırlar, gürültüyü bastırır.
      tileGridSize: görüntüyü kaç bölgeye bölerek işleneceği.

    Döner: (işlenmiş_rgb, clahe_uygulandı_mı)
    """
    if brightness >= LOW_LIGHT_BRIGHTNESS:
        return rgb, False  # yeterince parlak, değiştirme

    # RGB → LAB dönüşümü (L=parlaklık, A ve B renk kanalları)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Sadece L kanalına CLAHE uygula
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l2 = clahe.apply(l)

    # L kanalını geri birleştir, LAB → RGB'ye çevir
    enhanced = cv2.merge((l2, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB), True


# ---------------------------------------------------------------------------
# Aşama 5 — Grid montajı
# ---------------------------------------------------------------------------

def _build_grid(frames_rgb: list[np.ndarray]) -> np.ndarray:
    """Seçilen kareleri tek bir ızgara (grid) görseline birleştirir.

    NEDEN GRİD?
      VLM'e 8 ayrı görüntü yerine tek bir grid görseli gönderilir:
        - Tek HTTP/inference çağrısı → daha hızlı
        - Model tüm zaman dilimini bir arada görür → bağlamsal analiz daha iyi
        - Grid düzeni: GRID_COLS sütun × GRID_ROWS satır

    Eksik kare varsa (video çok kısaysa) siyah hücrelerle doldurulur.

    Çıktı boyutu: (CELL_SIZE[1]*GRID_ROWS) × (CELL_SIZE[0]*GRID_COLS) piksel
    Örnek: 384×384 hücre, 2×4 grid → 768×1536 piksel toplam grid
    """
    # Her kareyi CELL_SIZE'a yeniden boyutlandır (INTER_AREA: küçültme için ideal)
    cells = [cv2.resize(f, CELL_SIZE, interpolation=cv2.INTER_AREA) for f in frames_rgb]

    # Eksik hücreleri siyah ile doldur
    while len(cells) < GRID_ROWS * GRID_COLS:
        cells.append(np.zeros((CELL_SIZE[1], CELL_SIZE[0], 3), dtype=np.uint8))

    # Satırları yatay birleştir, sonra satırları dikey birleştir
    rows = [np.hstack(cells[r * GRID_COLS:(r + 1) * GRID_COLS]) for r in range(GRID_ROWS)]
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Ana giriş noktası — pipeline.py bu fonksiyonu çağırır
# ---------------------------------------------------------------------------

def build_vlm_frame_packet(video_path: str, video_id: str, output_dir: str) -> VLMFramePacket:
    """Uçtan uca ön işleme: video dosyası → S1b sözleşme paketi.

    ADIMLAR:
      1. Aday havuzu oluştur (_collect_candidates)
      2. En iyi kareleri seç (_select_frames) — fallback_ids'i de al
      3. Her kareye CLAHE uygula (gerekiyorsa)
      4. Her kare için FrameMeta oluştur (selection_reason dahil)
      5. Kareleri grid'e diz (_build_grid)
      6. Grid JPEG'ini diske yaz
      7. VLMFramePacket (S1b) oluştur ve döndür

    selection_reason ataması (S7 değişikliği):
      "fallback_fill"  → aşama 3'te zaman ekseninden dolduruldu
      "scene_change"   → SSIM farkı >= 0.15 (sahne değişimi)
      "uniform_sample" → SSIM farkı < 0.15 (tekrarlı ama seçildi)
      "motion_peak"    → ileride optik akış eklendiğinde kullanılacak

    Çıktı: VLMFramePacket → backend.py'nin infer() metoduna gönderilir.
    """
    candidates = _collect_candidates(video_path)
    if not candidates:
        raise ValueError("Video karesi okunamadı (boş ya da bozuk dosya olabilir).")

    # S7: _select_frames artık fallback_ids de döndürüyor
    selected, fallback_ids = _select_frames(candidates)

    enhanced_rgb: list[np.ndarray] = []
    clahe_used_any = False
    frame_metas: list[FrameMeta] = []

    for i, c in enumerate(selected):
        # CLAHE uygula (karanlık ise), sonucu listeye ekle
        rgb, used_clahe = _apply_clahe_if_needed(c.frame.rgb, c.brightness)
        clahe_used_any = clahe_used_any or used_clahe
        enhanced_rgb.append(rgb)

        # S7: Seçilme nedenini doğru belirle (3 dalil karar)
        if id(c) in fallback_ids:
            reason = "fallback_fill"    # zaman ekseni doldurma ile geldi
        elif c.ssim_diff >= 0.15:
            reason = "scene_change"     # SSIM farkı yüksek → sahne değişimi
        else:
            reason = "uniform_sample"   # SSIM farkı düşük ama seçildi
        # "motion_peak" → optik akış eklendiğinde buraya eklenecek

        frame_metas.append(FrameMeta(
            frame_index=c.frame.frame_index,
            timestamp_sec=c.frame.timestamp_sec,
            grid_position=i,           # grid'deki pozisyonu (0=sol üst)
            selection_reason=reason,
            quality=FrameQualityMetrics(
                laplacian_var=c.laplacian_var,
                ssim_diff=c.ssim_diff,
                brightness_mean=c.brightness,
            ),
        ))

    # Grid görselini oluştur ve diske yaz (JPEG, %92 kalite)
    grid_img = _build_grid(enhanced_rgb)
    os.makedirs(output_dir, exist_ok=True)
    packet_id = str(uuid.uuid4())  # benzersiz paket kimliği
    grid_path = os.path.join(output_dir, f"{packet_id}_grid.jpg")

    # RGB → BGR dönüşümü gerekli: cv2.imwrite BGR formatında yazar
    cv2.imwrite(grid_path, cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    # S1b paketini oluştur ve döndür → backend.py bunu alacak
    return VLMFramePacket(
        video_id=video_id,
        source_start_sec=selected[0].frame.timestamp_sec,   # seçilen ilk kare
        source_end_sec=selected[-1].frame.timestamp_sec,     # seçilen son kare
        frames=frame_metas,
        grid_layout=GridLayout(rows=GRID_ROWS, cols=GRID_COLS, cell_size=CELL_SIZE),
        enhancement=EnhancementInfo(
            clahe_applied=clahe_used_any,
            clip_limit=CLAHE_CLIP_LIMIT,
            tile_grid_size=CLAHE_TILE_GRID,
        ),
        grid_image_path=grid_path,
        packet_id=packet_id,
    )


# ===========================================================================
# VİDEO MODU ÖN İŞLEME — sahne sınırı bulma + 720p/60sn segment üretimi
# ===========================================================================
#
# NEDEN İKİNCİ BİR ÖN İŞLEME YOLU?
#   Yukarıdaki grid yolu (build_vlm_frame_packet) videoyu 8 kareye indirger.
#   Bu, görüntü-only modeller için gerekliydi. Buna karşılık TEKNOFEST EVREN
#   servisinde video analizine özelleşmiş bir model (alias "vlm") bulunur ve
#   videoyu doğrudan kabul eder (2.0 fps örnekleme, 520 kareye kadar). Videoyu
#   bütün olarak göndermek, 8 karelik mozaikten belirgin şekilde daha iyi
#   zamansal bağlam sağlar.
#
# İKİ SERT KISIT:
#   1. Çözünürlük: 720p. Kodlayıcının piksel bütçesi videonun TAMAMI için tek
#      bir toplamdır; 720p yükleme 77 saniyeden sonra orantılı olarak
#      küçültülür. Yani 720p üstüne çıkmak kazanç sağlamaz.
#   2. Süre: segment başına en fazla 60 saniye. Daha uzun videolar segmentlere
#      bölünüp SIRAYLA incelenir; segmentler arası bağlam metin tabanlı
#      hafızayla (contracts.VideoAnalysisMemory) taşınır. Böylece bağlam
#      penceresi segment sayısıyla büyümez.

from fractions import Fraction

import av

from contracts import format_mmss

# Sahne değişimi eşiği: 1-SSIM bu değeri aşarsa kesim adayı sayılır.
SCENE_SSIM_THRESHOLD = _CFG.get("scene_ssim_threshold", 0.30)
# Bir segment en az bu kadar uzun olmalı (çok kısa segment bağlamsız kalır).
SCENE_MIN_SEGMENT_SEC = _CFG.get("scene_min_segment_sec", 15.0)
# Segment üst sınırı — EVREN video kısıtı (720p için 60 sn).
SCENE_MAX_SEGMENT_SEC = _CFG.get("scene_max_segment_sec", 60.0)
# SSIM her kaç karede bir hesaplanır (hız/hassasiyet dengesi).
SCENE_DETECT_STRIDE = _CFG.get("scene_detect_stride", 5)
# Segment klipleri bu yüksekliğe indirilir (720p kısıtı).
VLM_MAX_HEIGHT = _CFG.get("vlm_max_height", 720)
# x264 kalite/hız ayarı — kalite ile dosya boyutu dengesi.
VLM_CLIP_CRF = _CFG.get("vlm_clip_crf", 23)
VLM_CLIP_PRESET = _CFG.get("vlm_clip_preset", "veryfast")


@dataclass
class VideoClip:
    """VLM'e gönderilmeye hazır tek bir video parçası.

    ``reencoded=False`` ise dosya özgün videonun kendisidir (kısıtlara zaten
    uyduğu için yeniden kodlanmamıştır); bu durumda dosya **silinmemelidir**.
    """
    path: str                  # gönderilecek mp4 dosyasının yolu
    start_sec: float           # tam videodaki mutlak başlangıç
    end_sec: float             # tam videodaki mutlak bitiş
    index: int = 0             # segment sırası (0'dan başlar)
    reencoded: bool = False    # geçici dosya mı (True ise temizlenebilir)

    @property
    def duration_sec(self) -> float:
        """Klip süresi (saniye)."""
        return max(0.0, self.end_sec - self.start_sec)

    @property
    def time_label(self) -> str:
        """İnsan okunur ``MM:SS-MM:SS`` etiketi."""
        return f"{format_mmss(self.start_sec)}-{format_mmss(self.end_sec)}"


def probe_video(video_path: str) -> dict:
    """Videonun temel özelliklerini okur (kod çözmeden, container meta verisinden).

    Args:
        video_path: İncelenecek video dosyası.

    Returns:
        ``width``, ``height``, ``fps``, ``duration_sec`` anahtarlı sözlük.
        Süre meta veriden okunamazsa kare sayısı/fps ile tahmin edilir.
    """
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 25.0

        duration = 0.0
        if stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        elif container.duration:
            duration = float(container.duration) / av.time_base
        if not duration and stream.frames and fps:
            duration = stream.frames / fps

        return {
            "width": int(stream.codec_context.width),
            "height": int(stream.codec_context.height),
            "fps": fps,
            "duration_sec": float(duration),
        }


def detect_scene_boundaries(
    video_path: str,
    ssim_threshold: float | None = None,
    min_segment_sec: float | None = None,
    max_segment_sec: float | None = None,
    stride: int | None = None,
) -> list[tuple[float, float]]:
    """Videoyu sahne değişimlerinden yararlanarak segment aralıklarına böler.

    Kesim noktaları rastgele değil, görüntünün gerçekten değiştiği anlardır;
    böylece bir olay iki segmentin ortasından bölünme olasılığı azalır. Ancak
    sahne değişimi beklenirken segmentin sonsuza uzamasına izin verilmez:
    ``max_segment_sec`` sert bir tavandır (EVREN'in video kısıtı).

    Args:
        video_path: Bölünecek video.
        ssim_threshold: ``1-SSIM`` bu değeri aşarsa sahne değişimi sayılır.
            ``None`` ise config değeri kullanılır.
        min_segment_sec: Bir segmentin asgari süresi; bu süreden önce gelen
            sahne değişimleri kesim için kullanılmaz.
        max_segment_sec: Segment üst sınırı; sahne değişimi bulunmasa da bu
            sürede kesilir.
        stride: SSIM'in kaç karede bir hesaplanacağı.

    Returns:
        ``(start_sec, end_sec)`` ikilileri; boşluksuz ve sıralı olarak videonun
        tamamını kaplar. Video zaten sınırın altındaysa tek elemanlı liste.
    """
    ssim_threshold = SCENE_SSIM_THRESHOLD if ssim_threshold is None else ssim_threshold
    min_segment_sec = SCENE_MIN_SEGMENT_SEC if min_segment_sec is None else min_segment_sec
    max_segment_sec = SCENE_MAX_SEGMENT_SEC if max_segment_sec is None else max_segment_sec
    stride = SCENE_DETECT_STRIDE if stride is None else stride

    duration = probe_video(video_path)["duration_sec"]
    if duration <= 0:
        raise ValueError(f"Video süresi belirlenemedi: {video_path}")

    # Tek segmente sığıyorsa video hiç taranmaz (gereksiz iş yapılmaz).
    if duration <= max_segment_sec:
        return [(0.0, duration)]

    # Sahne değişimi anlarını topla.
    changes: list[float] = []
    prev_gray = None
    for frame in read_video_stream(video_path):
        if frame.frame_index % stride != 0:
            continue
        gray = _to_gray_small(frame.rgb)
        if prev_gray is not None:
            if float(1.0 - ssim(prev_gray, gray)) >= ssim_threshold:
                changes.append(frame.timestamp_sec)
        prev_gray = gray

    # Zaman eksenini yürüyerek segmentleri kur.
    segments: list[tuple[float, float]] = []
    start = 0.0
    cursor = 0
    while start < duration - 1e-6:
        hard_end = min(start + max_segment_sec, duration)

        # min_segment_sec'ten önceki değişimleri atla (çok kısa segment olmasın).
        while cursor < len(changes) and changes[cursor] <= start + min_segment_sec:
            cursor += 1

        if cursor < len(changes) and changes[cursor] < hard_end:
            end = changes[cursor]
            cursor += 1
        else:
            end = hard_end

        segments.append((start, end))
        start = end

    # Çok kısa kalan son segmenti öncekine kat — ama YALNIZCA üst sınır
    # aşılmıyorsa; 60 sn kısıtı pazarlık konusu değildir.
    if len(segments) >= 2:
        last_start, last_end = segments[-1]
        if (last_end - last_start) < min_segment_sec:
            prev_start, _ = segments[-2]
            if (last_end - prev_start) <= max_segment_sec:
                segments[-2] = (prev_start, last_end)
                segments.pop()

    return segments


def extract_video_segment(
    video_path: str,
    start_sec: float,
    end_sec: float,
    out_path: str,
    max_height: int | None = None,
) -> str:
    """Videonun bir aralığını kesip 720p'ye indirerek yeni bir mp4 yazar.

    Yeniden kodlama gerekir çünkü hem kırpma hem ölçekleme yapılır; akış
    kopyalama (remux) yalnızca anahtar kare sınırlarında kesebildiği için
    istenen başlangıç anını tutturamaz.

    Args:
        video_path: Kaynak video.
        start_sec: Kesimin başlangıcı (saniye).
        end_sec: Kesimin bitişi (saniye).
        out_path: Yazılacak mp4 dosyası.
        max_height: Hedef azami yükseklik; ``None`` ise config değeri (720).

    Returns:
        Yazılan dosyanın yolu (``out_path``).

    Raises:
        ValueError: Aralıkta hiç kare bulunamazsa.
    """
    max_height = VLM_MAX_HEIGHT if max_height is None else max_height
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with av.open(video_path) as inp:
        istream = inp.streams.video[0]
        istream.thread_type = "AUTO"

        in_w = int(istream.codec_context.width)
        in_h = int(istream.codec_context.height)
        rate = istream.average_rate or Fraction(25, 1)
        time_base = float(istream.time_base) if istream.time_base else 1.0

        # 720p'ye indir; yuv420p çift boyut gerektirdiği için aşağı yuvarla.
        scale = min(1.0, max_height / in_h) if in_h else 1.0
        out_w = max(2, (int(in_w * scale) // 2) * 2)
        out_h = max(2, (int(in_h * scale) // 2) * 2)

        # Başlangıca yaklaş (anahtar kareye), kalan farkı kare atlayarak kapat.
        if start_sec > 0 and time_base:
            try:
                inp.seek(int(start_sec / time_base), stream=istream)
            except av.AVError:
                inp.seek(0)

        out = av.open(out_path, mode="w")
        try:
            ostream = out.add_stream("libx264", rate=rate)
            ostream.width = out_w
            ostream.height = out_h
            ostream.pix_fmt = "yuv420p"
            ostream.options = {"crf": str(VLM_CLIP_CRF), "preset": VLM_CLIP_PRESET}
            ostream.time_base = Fraction(1, 90000)

            written = 0
            for frame in inp.decode(istream):
                if frame.pts is None:
                    continue
                t = float(frame.pts) * time_base
                if t < start_sec - 1e-3:
                    continue
                if t >= end_sec:
                    break

                new_frame = frame.reformat(width=out_w, height=out_h, format="yuv420p")
                # Zaman damgasını segment başlangıcına göre sıfırla; klip kendi
                # içinde 0'dan başlar, mutlak zaman prompt'ta ayrıca bildirilir.
                new_frame.pts = int(round((t - start_sec) / float(ostream.time_base)))
                new_frame.time_base = ostream.time_base

                for packet in ostream.encode(new_frame):
                    out.mux(packet)
                written += 1

            for packet in ostream.encode():  # kodlayıcıyı boşalt
                out.mux(packet)
        finally:
            out.close()

    if written == 0:
        raise ValueError(
            f"{start_sec:.1f}-{end_sec:.1f} sn aralığında kare bulunamadı: {video_path}"
        )
    return out_path


def prepare_video_segments(
    video_path: str,
    output_dir: str,
    max_segment_sec: float | None = None,
    max_height: int | None = None,
    ssim_threshold: float | None = None,
    min_segment_sec: float | None = None,
) -> list[VideoClip]:
    """Videoyu EVREN ``vlm`` modeline gönderilmeye hazır kliplere dönüştürür.

    Karar mantığı:

    * Video hem süre hem çözünürlük kısıtına **zaten uyuyorsa** yeniden
      kodlanmaz; özgün dosya olduğu gibi kullanılır. Bu hem zaman kazandırır
      hem de ön ek önbelleği (prefix cache) açısından önemlidir: baytlar
      değişmediği sürece aynı video üzerinden sorulan takip soruları
      önbellekten yararlanır.
    * Aksi hâlde video :func:`detect_scene_boundaries` ile segmentlere bölünür
      ve her segment 720p'ye indirilerek ayrı bir mp4 olarak yazılır.

    Args:
        video_path: Kaynak video.
        output_dir: Segment kliplerinin yazılacağı klasör.
        max_segment_sec: Segment süre tavanı (varsayılan 60).
        max_height: Çözünürlük tavanı (varsayılan 720).
        ssim_threshold: Sahne değişimi eşiği.
        min_segment_sec: Segment asgari süresi.

    Returns:
        Zaman sırasında :class:`VideoClip` listesi. Tek parça yeterliyse tek
        elemanlı ve ``reencoded=False`` olur.
    """
    max_segment_sec = SCENE_MAX_SEGMENT_SEC if max_segment_sec is None else max_segment_sec
    max_height = VLM_MAX_HEIGHT if max_height is None else max_height

    info = probe_video(video_path)
    duration = info["duration_sec"]
    if duration <= 0:
        raise ValueError(f"Video süresi belirlenemedi: {video_path}")

    fits_duration = duration <= max_segment_sec
    fits_resolution = info["height"] <= max_height
    if fits_duration and fits_resolution:
        return [VideoClip(
            path=video_path, start_sec=0.0, end_sec=duration, index=0, reencoded=False
        )]

    ranges = detect_scene_boundaries(
        video_path,
        ssim_threshold=ssim_threshold,
        min_segment_sec=min_segment_sec,
        max_segment_sec=max_segment_sec,
    )

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]

    clips: list[VideoClip] = []
    for idx, (start, end) in enumerate(ranges):
        out_path = os.path.join(output_dir, f"{stem}_seg{idx:03d}.mp4")
        extract_video_segment(video_path, start, end, out_path, max_height=max_height)
        clips.append(VideoClip(
            path=out_path, start_sec=start, end_sec=end, index=idx, reencoded=True
        ))
    return clips
