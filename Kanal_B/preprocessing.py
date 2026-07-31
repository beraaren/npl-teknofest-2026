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