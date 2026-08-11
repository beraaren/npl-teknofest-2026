"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DOSYA: pipeline.py                                                          ║
║  KATMAN: Orchestrator — Kanal B'nin uçtan uca çalıştırıcısı                 ║
║  ROL   : Diğer modülleri bir araya getirir; Karar Ajanı bu dosyayı çağırır. ║
╚══════════════════════════════════════════════════════════════════════════════╝

VERİ AKIŞI İÇİNDEKİ YERİ:
  Karar Ajanı → run_channel_b(video_path, video_id) →
    preprocessing.build_vlm_frame_packet()  [S1b üret]
    → backend.build_backend(VLM_CONFIG)     [backend seç]
    → backend.infer(packet)                  [VLM çağrısı]
    → S8 dict                               → Karar Ajanı'na ilet

KRİTİK DÜZELTMELERİN ÖZETİ:
  1. Import isimleri düzeltildi:
     - Eski: `from channel_b_preprocessing import ...` → ImportError veriyordu
     - Eski: `from vlm_backend import ...`             → ImportError veriyordu
     - Yeni: gerçek modül isimleri (preprocessing, backend)

  2. VLM_CONFIG artık hardcode değil:
     - Eski: modül seviyesinde sabit dict
     - Yeni: _load_vlm_config() → config.yaml'dan okunur, yoksa fallback

CONFIG.YAML OKUMA MANTIĞI (_load_vlm_config):
  config.yaml'da [vlm.default_backend] değeri okunur.
  "auto" → [vlm.auto_preference] listesinin ilk elemanı seçilir.
  Seçilen backend'in ayarları (model, max_tokens, temperature) alınır.
  Tüm bunlar build_backend()'in beklediği dict formatına dönüştürülür.
"""
import json
import os

# KRİTİK DÜZELTME: Eski yanlış import isimleri → gerçek modül isimleri
from preprocessing import build_vlm_frame_packet, detect_scene_boundaries  # eskisi: channel_b_preprocessing
from backend import build_backend                   # eskisi: vlm_backend
from contracts import SegmentResult, BatchResult    # batch işleme sözleşmeleri


# ---------------------------------------------------------------------------
# Config yükleme — VLM backend ayarlarını config.yaml'dan oku
# ---------------------------------------------------------------------------

def _load_vlm_config() -> dict:
    """Proje kökündeki config.yaml'dan VLM backend konfigürasyonunu yükler.

    NEDEN FONKSİYON?
      Daha önce VLM_CONFIG modül seviyesinde hardcode edilmişti.
      Şimdi config.yaml'dan okunuyor → model değiştirmek için kodu açmak gerekmez.

    DÖNER:
      build_backend() fonksiyonunun beklediği dict:
        {backend, base_url, model_name, max_tokens} veya
        {backend, model_name, device, max_tokens}

    config.yaml BULUNAMAZSA:
      Güvenli fallback olarak llama.cpp + localhost:8080 kullanılır.
      Bu sayede config olmadan da sistem ayağa kalkar.
    """
    # Kanal_B/ klasöründen bir üst dizindeki config.yaml'a ulaş
    config_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    )

    if not os.path.exists(config_path):
        # config.yaml yoksa güvenli fallback — hiç sessiz çalış
        return {
            "backend": "llama.cpp",
            "base_url": "http://localhost:8080",
            "model_name": "llava-v1.6-mistral-7b",
            "max_tokens": 512,
        }

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        vlm_cfg = cfg.get("vlm", {})

        # "auto" backend → preference listesinin ilk kullanılabilir backend'i
        backend_choice = vlm_cfg.get("default_backend", "auto")
        if backend_choice == "auto":
            backend_choice = vlm_cfg.get("auto_preference", ["llama.cpp"])[0]

        # config.yaml key isimleri: "llama_cpp" ama backend_name: "llama.cpp"
        # replace(".", "_") → llama.cpp → llama_cpp dönüşümü
        backend_key      = backend_choice.replace(".", "_")
        backend_specific = vlm_cfg.get(backend_key, {})

        # build_backend() sözleşmesine uygun dict oluştur
        if backend_choice in ("vllm", "llama.cpp"):
            # Sunucu tabanlı backend'ler: base_url gerekli
            return {
                "backend":     backend_choice,
                "base_url":    backend_specific.get("base_url", "http://localhost:8080"),
                "model_name":  backend_specific.get("model", "unknown-model"),
                # vLLM config.yaml'da key "max_new_tokens", llama.cpp'de "max_tokens".
                # Her iki key de denenir; ikisi de yoksa 1024 default kullanılır.
                "max_tokens":  backend_specific.get("max_tokens",
                               backend_specific.get("max_new_tokens", 1024)),
                "temperature": backend_specific.get("temperature", 0.15),
                # [#5] HTTP timeout → config.yaml'dan (büyük modeller için artırılabilir)
                "timeout_sec": backend_specific.get("timeout_sec", 60),
            }
        else:
            # Transformers backend: base_url gerekmez, device lazım
            return {
                "backend":    "transformers",
                "model_name": backend_specific.get("model", "llava-hf/LLaVA-NeXT-Video-7B-hf"),
                "device":     "cuda",
                "max_tokens": backend_specific.get("max_new_tokens", 512),
            }

    except Exception as exc:
        # YAML parse hatası veya beklenmedik yapı → fallback ve devam et
        print(f"[pipeline] config.yaml okunamadı ({exc}), fallback config kullanılıyor.")
        return {
            "backend":    "llama.cpp",
            "base_url":   "http://localhost:8080",
            "model_name": "llava-v1.6-mistral-7b",
            "max_tokens": 512,
        }


# Modül yüklendiğinde config'i bir kez oku — tekrar çağrılmaz
VLM_CONFIG = _load_vlm_config()


# ---------------------------------------------------------------------------
# Kanal B ana giriş noktası — Karar Ajanı buraya bağlanır
# ---------------------------------------------------------------------------

def run_channel_b(video_path: str, video_id: str, output_dir: str = "./out/channel_b") -> dict:
    """Kanal B'nin uçtan uca çalıştırıcısı — video'dan S8 dict'e.

    PARAMETRELER:
      video_path : işlenecek video dosyasının tam yolu
      video_id   : izlenebilirlik için benzersiz kimlik — S1b ve S8 paketlerinde kullanılır
      output_dir : grid JPEG ve vlm_interpretation JSON'ın yazıldığı klasör

    DÖNER:
      S8 sözleşmesine uygun dict → Karar Ajanı'nın 3. girdisi
      Örnek alanlar: scene_summary_tr, detected_entities, risk_flags_tr...

    ADIMLAR:
      1. build_vlm_frame_packet() → video analiz + CLAHE + grid → S1b paketi
      2. build_backend(VLM_CONFIG) → config.yaml'a göre backend seç
      3. backend.infer(packet)    → VLM çağrısı → S8 paketi
      4. JSON olarak diske yaz   → debug/audit için
      5. S8 dict döndür          → Karar Ajanı alır
    """
    # Adım 1: ön işleme — video → S1b paketi (grid JPEG dahil)
    packet = build_vlm_frame_packet(video_path, video_id, output_dir)

    # Adım 2-3: backend seç ve VLM çağrısı yap → S8 paketi
    backend        = build_backend(VLM_CONFIG)
    interpretation = backend.infer(packet)

    # Adım 4: çıktıyı diske yaz (debug/audit — zorunlu değil ama önerilir)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{packet.packet_id}_vlm_interpretation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(interpretation.to_dict(), f, ensure_ascii=False, indent=2)

    # Adım 5: S8 sözleşme dict'ini döndür → Karar Ajanı'nın 3. girdisi
    return interpretation.to_dict()


# ---------------------------------------------------------------------------
# Sahne Bazlı Batch İşleme — uzun videolar için yeni giriş noktası
# ---------------------------------------------------------------------------

def run_channel_b_scene_based(
    video_path: str,
    video_id: str,
    output_dir: str = "./out/channel_b",
    ssim_threshold: float | None = None,
    min_segment_sec: float | None = None,
    max_segment_sec: float | None = None,
) -> BatchResult:
    """Uzun videolar için sahne değişimi bazlı batch işleme.

    Klasik run_channel_b() tüm videodan 8 kare seçer; uzun videolarda
    kritik olaylar temsil dışında kalabilir. Bu fonksiyon:
      1. detect_scene_boundaries() ile video sahne sınırlarından otomatik bölünür
      2. Her sahne segmenti için ayrı VLM çağrısı yapılır (8 kare / segment)
      3. Tüm sonuçlar BatchResult olarak Karar Ajanı'na iletilir

    PARAMETRELER:
      video_path      : İşlenecek video dosyasının tam yolu
      video_id        : İzlenebilirlik için benzersiz kimlik
      output_dir      : Grid JPEG ve JSON çıktılarının kaydedileceği klasör
      ssim_threshold  : Sahne değişimi eşiği (None → config.yaml default: 0.30)
      min_segment_sec : Minimum segment süresi sn (None → config.yaml default: 15.0)
      max_segment_sec : Maximum segment süresi sn (None → config.yaml default: 60.0)

    DÖNER:
      BatchResult — tüm segment sonuçları + özet istatistikler
      BatchResult.most_critical_segment → en yüksek severity'li segment
      BatchResult.segments              → zaman sırasında tüm SegmentResult'lar

    GERİYE DÖNÜŞ UYUMLULUĞU:
      run_channel_b() hiç değişmedi — kısa videolar için eski kullanım çalışmaya devam eder.
    """
    # detect_scene_boundaries() parametreleri: None verilirse preprocessing.py default'ları kullanılır
    detect_kwargs: dict = {}
    if ssim_threshold is not None:
        detect_kwargs["ssim_threshold"] = ssim_threshold
    if min_segment_sec is not None:
        detect_kwargs["min_segment_sec"] = min_segment_sec
    if max_segment_sec is not None:
        detect_kwargs["max_segment_sec"] = max_segment_sec

    segments_ranges = detect_scene_boundaries(video_path, **detect_kwargs)
    print(f"[pipeline] {len(segments_ranges)} sahne segmenti işlenecek: {video_id}")

    backend = build_backend(VLM_CONFIG)
    os.makedirs(output_dir, exist_ok=True)

    segment_results: list[SegmentResult] = []
    failed_segments: list[dict] = []   # [#3] başarısız segmentleri izle

    for idx, (start_sec, end_sec) in enumerate(segments_ranges):
        print(
            f"[pipeline] Segment {idx + 1}/{len(segments_ranges)}: "
            f"{start_sec:.1f}sn – {end_sec:.1f}sn"
        )
        try:
            # Her segment için S1b paketi oluştur (zaman aralığı sınırlı)
            packet = build_vlm_frame_packet(
                video_path, video_id, output_dir,
                start_sec=start_sec, end_sec=end_sec,
            )

            # VLM çağrısı → S8 paketi
            interpretation = backend.infer(packet)

            # Sonucu diske yaz (debug/audit)
            out_path = os.path.join(
                output_dir,
                f"{packet.packet_id}_seg{idx:03d}_vlm_interpretation.json",
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(interpretation.to_dict(), f, ensure_ascii=False, indent=2)

            segment_results.append(SegmentResult(
                segment_index=idx,
                start_sec=start_sec,
                end_sec=end_sec,
                interpretation=interpretation,
            ))

        except Exception as exc:
            # [#3] Tek bir segment'in başarısız olması diğerlerini durdurmamalı.
            # Sessiz veri kaybı yerine failed_segments'e ekle → Karar Ajanı bilgilendirilir.
            err_msg = f"{type(exc).__name__}: {exc}"
            print(f"[pipeline] UYARI: Segment {idx} işlenemedi ({err_msg}), atlanıyor.")
            failed_segments.append({
                "segment_index": idx,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "error": err_msg,
            })

    result = BatchResult(video_id=video_id, segments=segment_results, failed_segments=failed_segments)
    print(
        f"[pipeline] Tamamlandı: {result.total_segments} segment, "
        f"{result.total_risk_events} risk olayı, "
        f"en kritik segment: "
        f"{result.most_critical_segment.segment_index if result.most_critical_segment else 'yok'}"
    )
    return result


# ---------------------------------------------------------------------------
# Doğrudan çalıştırma — hızlı test için
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Örnek kullanım: python pipeline.py
    # video.mp4 proje kökünde olmalı
    out = run_channel_b("video.mp4", video_id="demo-001")
    print(json.dumps(out, ensure_ascii=False, indent=2))