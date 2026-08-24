"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DOSYA: pipeline.py                                                          ║
║  KATMAN: Orchestrator — Kanal B'nin uçtan uca çalıştırıcısı                 ║
║  ROL   : Diğer modülleri bir araya getirir; Karar Ajanı bu dosyayı çağırır. ║
╚══════════════════════════════════════════════════════════════════════════════╝

VERİ AKIŞI İÇİNDEKİ YERİ:
  Karar Ajanı → run_channel_b(video_path, video_id) →
    preprocessing.prepare_video_segments()   [720p / 60 sn klipler]
    → backend.build_backend(VLM_CONFIG)      [EVREN server backend]
    → backend.infer_video(clip)              [her klip için bir çağrı]
    → S8 dict                                → Karar Ajanı'na ilet

İKİ ÇALIŞMA MODU — video uzunluğuna göre otomatik seçilir
---------------------------------------------------------
1. **Tek parça** (video ≤ 60 sn ve ≤ 720p): video olduğu gibi tek çağrıda
   gönderilir. Yeniden kodlama yapılmadığı için baytlar sabit kalır ve aynı
   klip üzerinden sorulan takip soruları ön ek önbelleğinden (prefix cache)
   yararlanır.

2. **İterasyonlu** (daha uzun videolar): video sahne sınırlarından 60 saniyeyi
   aşmayan segmentlere bölünür ve segmentler SIRAYLA incelenir. Her segmentin
   yorumu, **metin tabanlı** anlamsal + zamansal hafızaya damıtılır
   (:class:`contracts.VideoAnalysisMemory`) ve bir sonraki segmentin prompt'una
   eklenir.

   Neden metin hafıza? Segmentlerin ham çıktılarını biriktirmek bağlam
   penceresini segment sayısıyla doğrusal şişirir ve uzun videolarda taşmaya
   yol açar. Damıtılmış metin satırları üst sınırla kırpıldığı için bağlam
   maliyeti sabit bir tavana oturur; buna karşılık süregelen durumlar
   (aynı kişi, devam eden risk) segmentler arasında korunur.

   Zaman ekseni: her segmente kendi **mutlak** başlangıç saniyesi bildirilir,
   böylece olay zaman damgaları klip göreli değil tam video göreli üretilir.
"""
import json
import os

# KRİTİK DÜZELTME: Eski yanlış import isimleri → gerçek modül isimleri
from preprocessing import (             # eskisi: channel_b_preprocessing
    build_vlm_frame_packet,
    detect_scene_boundaries,
    prepare_video_segments,
    probe_video,
)
# NOT: Bu modül bilinçli olarak "vlm_backend" adını taşır, "backend" DEĞİL.
# Depo kökünde mikroservisleri barındıran bir `backend/` paketi var; Kanal_B
# dizini sys.path'e eklendiğinde `from backend import ...` o pakete çözülüp
# ImportError veriyordu (mikroservis bağlamında `backend` zaten sys.modules'te).
from vlm_backend import build_backend
from contracts import (
    SegmentResult, BatchResult, VideoAnalysisMemory,
)


# ---------------------------------------------------------------------------
# Config yükleme — VLM backend ayarlarını config.yaml'dan oku
# ---------------------------------------------------------------------------

def _load_vlm_config() -> dict:
    """Proje kökündeki config.yaml'dan VLM backend konfigürasyonunu yükler.

    NEDEN FONKSİYON?
      Daha önce VLM_CONFIG modül seviyesinde hardcode edilmişti.
      Şimdi config.yaml'dan okunuyor → model değiştirmek için kodu açmak gerekmez.

    DESTEKLENEN BACKEND'LER:
      ``server``       → OpenAI-uyumlu harici servis (TEKNOFEST EVREN). Varsayılan.
      ``vllm``         → yerel vllm serve (HTTP)
      ``llama.cpp``    → yerel llama-server (HTTP)
      ``transformers`` → süreç içi model

    DÖNER:
      build_backend() fonksiyonunun beklediği dict.

    config.yaml BULUNAMAZSA:
      EVREN varsayılanlarına düşer; anahtar ortam değişkeninden okunur.
    """
    fallback = {
        "backend": "server",
        "base_url": "https://evren-llmapi.ssyz.org.tr/v1",
        "model_name": "llm-large",
        "video_model": "vlm",
        "api_key_env": "EVREN_API_KEY",
        "max_tokens": 65536,
        "temperature": 0.15,
        "timeout_sec": 1800,
        "enable_thinking": False,
    }

    config_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    )
    if not os.path.exists(config_path):
        return fallback

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        vlm_cfg = cfg.get("vlm", {})

        # "auto" backend → preference listesinin ilk elemanı
        backend_choice = vlm_cfg.get("default_backend", "server")
        if backend_choice == "auto":
            backend_choice = (vlm_cfg.get("auto_preference") or ["server"])[0]

        # config.yaml key isimleri alt çizgili ("llama_cpp"), backend adı
        # noktalı olabilir ("llama.cpp") → dönüşüm
        backend_key = backend_choice.replace(".", "_")
        spec = vlm_cfg.get(backend_key, {}) or {}

        if backend_choice == "server":
            return {
                "backend": "server",
                "base_url": spec.get("base_url", fallback["base_url"]),
                "model_name": spec.get("model_name", fallback["model_name"]),
                "video_model": spec.get("video_model", fallback["video_model"]),
                "api_key_env": spec.get("api_key_env", fallback["api_key_env"]),
                "max_tokens": spec.get("max_tokens", fallback["max_tokens"]),
                "temperature": spec.get("temperature", fallback["temperature"]),
                "timeout_sec": spec.get("timeout_sec", fallback["timeout_sec"]),
                "enable_thinking": spec.get("enable_thinking", False),
            }

        if backend_choice in ("vllm", "llama.cpp"):
            return {
                "backend": backend_choice,
                "base_url": spec.get("base_url", "http://localhost:8080"),
                "model_name": spec.get("model", "unknown-model"),
                "max_tokens": spec.get("max_tokens", spec.get("max_new_tokens", 1024)),
                "temperature": spec.get("temperature", 0.15),
                "timeout_sec": spec.get("timeout_sec", 1800),
            }

        if backend_choice == "transformers":
            return {
                "backend": "transformers",
                "model_name": spec.get("model", "llava-hf/LLaVA-NeXT-Video-7B-hf"),
                "device": spec.get("device", "cuda"),
                "max_tokens": spec.get("max_new_tokens", 512),
            }

        # Tanınmayan ad: sessizce yanlış modele düşmek yerine EVREN'e dön.
        print(f"[pipeline] Bilinmeyen backend '{backend_choice}'; EVREN'e dönülüyor.")
        return fallback

    except Exception as exc:
        print(f"[pipeline] config.yaml okunamadı ({exc}), EVREN fallback kullanılıyor.")
        return fallback


# Modül yüklendiğinde config'i bir kez oku
VLM_CONFIG = _load_vlm_config()


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def _write_json(obj: dict, path: str) -> None:
    """Sözlüğü UTF-8 JSON olarak diske yazar (debug/audit izi)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _run_grid_fallback(backend, video_path: str, video_id: str, output_dir: str) -> dict:
    """Video modunu desteklemeyen backend'ler için eski grid yolunu çalıştırır.

    Yerel backend'ler (``vllm``, ``llama.cpp``, ``transformers``) videoyu
    doğrudan kabul etmez; onlar için kareler tek bir mozaikte birleştirilip
    görüntü olarak gönderilir.
    """
    print("[pipeline] Backend video modunu desteklemiyor; grid yoluna düşülüyor.")
    packet = build_vlm_frame_packet(video_path, video_id, output_dir)
    interpretation = backend.infer(packet)
    _write_json(
        interpretation.to_dict(),
        os.path.join(output_dir, f"{packet.packet_id}_vlm_interpretation.json"),
    )
    return interpretation.to_dict()


# ---------------------------------------------------------------------------
# Kanal B ana giriş noktası — Karar Ajanı buraya bağlanır
# ---------------------------------------------------------------------------

def run_channel_b(video_path: str, video_id: str, output_dir: str = "./out/channel_b") -> dict:
    """Kanal B'nin uçtan uca çalıştırıcısı — video'dan S8 dict'e.

    Video uzunluğuna göre iki moddan biri otomatik seçilir (bkz. modül
    docstring'i): tek parça veya segment segment iterasyon.

    PARAMETRELER:
      video_path : işlenecek video dosyasının tam yolu
      video_id   : izlenebilirlik için benzersiz kimlik
      output_dir : segment klipleri, JSON çıktıları ve grid'in yazıldığı klasör

    DÖNER:
      S8 sözleşmesine uygun dict → Karar Ajanı'nın 3. girdisi.
      İterasyonlu modda segmentler tek yoruma indirgenir
      (:meth:`contracts.BatchResult.to_interpretation_dict`) ve ek olarak
      ``segment_count`` / ``failed_segments`` / ``memory_context`` alanları
      taşınır; böylece karar ajanı analizin kaç parçadan geldiğini bilir.
    """
    backend = build_backend(VLM_CONFIG)
    os.makedirs(output_dir, exist_ok=True)

    # Video modunu desteklemeyen (yerel) backend'ler için eski grid yolu
    if not hasattr(backend, "infer_video"):
        return _run_grid_fallback(backend, video_path, video_id, output_dir)

    info = probe_video(video_path)
    clips = prepare_video_segments(video_path, os.path.join(output_dir, "segments"))

    # --- Mod 1: tek parça -------------------------------------------------
    if len(clips) == 1:
        clip = clips[0]
        print(
            f"[pipeline] Tek parça mod: {info['width']}x{info['height']}, "
            f"{info['duration_sec']:.1f}sn (yeniden kodlama: {clip.reencoded})"
        )
        interpretation = backend.infer_video(
            video_path=clip.path,
            video_id=video_id,
            clip_start_sec=clip.start_sec,
            clip_end_sec=clip.end_sec,
            is_segment=False,
        )
        _write_json(
            interpretation.to_dict(),
            os.path.join(output_dir, f"{interpretation.packet_id}_vlm_interpretation.json"),
        )
        return interpretation.to_dict()

    # --- Mod 2: iterasyonlu ----------------------------------------------
    print(
        f"[pipeline] İterasyonlu mod: {info['duration_sec']:.1f}sn video "
        f"{len(clips)} segmente bölündü (segment tavanı 60sn / 720p)"
    )

    memory = VideoAnalysisMemory()
    segments: list[SegmentResult] = []
    failed: list[dict] = []

    for clip in clips:
        print(f"[pipeline]   segment {clip.index + 1}/{len(clips)}: {clip.time_label}")
        try:
            interpretation = backend.infer_video(
                video_path=clip.path,
                video_id=video_id,
                clip_start_sec=clip.start_sec,
                clip_end_sec=clip.end_sec,
                # Önceki segmentlerin damıtılmış metin bağlamı
                memory_context=memory.to_prompt_context(),
                is_segment=True,
            )
        except Exception as exc:
            # Tek segmentin başarısızlığı analizi durdurmamalı; sessiz veri
            # kaybı yerine failed_segments'e yazılır ve karar ajanı bilir.
            err = f"{type(exc).__name__}: {exc}"
            print(f"[pipeline]   UYARI: segment {clip.index} işlenemedi ({err})")
            failed.append({
                "segment_index": clip.index,
                "start_sec": clip.start_sec,
                "end_sec": clip.end_sec,
                "error": err,
            })
            continue

        # Hafızayı güncelle → sonraki segment bu bağlamla yorumlanır
        memory.absorb(interpretation, clip.start_sec)

        segments.append(SegmentResult(
            segment_index=clip.index,
            start_sec=clip.start_sec,
            end_sec=clip.end_sec,
            interpretation=interpretation,
        ))
        _write_json(
            interpretation.to_dict(),
            os.path.join(
                output_dir,
                f"{interpretation.packet_id}_seg{clip.index:03d}_vlm_interpretation.json",
            ),
        )

    batch = BatchResult(
        video_id=video_id,
        segments=segments,
        failed_segments=failed,
        memory_context=memory.to_prompt_context(),
    )
    _write_json(batch.to_dict(), os.path.join(output_dir, f"{video_id}_batch.json"))

    critical = batch.most_critical_segment
    print(
        f"[pipeline] Tamamlandı: {batch.total_segments} segment, "
        f"{batch.total_risk_events} risk olayı, "
        f"{len(failed)} başarısız, en kritik segment: "
        f"{critical.segment_index if critical else 'yok'}"
    )
    return batch.to_interpretation_dict()


# ---------------------------------------------------------------------------
# Sahne Bazlı Batch İşleme — tam BatchResult isteyen çağıranlar için
# ---------------------------------------------------------------------------

def run_channel_b_scene_based(
    video_path: str,
    video_id: str,
    output_dir: str = "./out/channel_b",
    ssim_threshold: float | None = None,
    min_segment_sec: float | None = None,
    max_segment_sec: float | None = None,
) -> BatchResult:
    """Segment segment analiz eder ve **tam** :class:`BatchResult` döndürür.

    :func:`run_channel_b` ile aynı iterasyon mantığını kullanır; farkı,
    sonucu tek yoruma indirgemeyip segment ayrıntılarını (her segmentin kendi
    yorumu, en kritik segment, başarısızlıklar) koruyarak döndürmesidir. Segment
    bazlı raporlama veya UI zaman çizelgesi gerektiğinde bu kullanılır.

    PARAMETRELER:
      ssim_threshold  : Sahne değişimi eşiği (None → config: 0.30)
      min_segment_sec : Minimum segment süresi (None → config: 15.0)
      max_segment_sec : Maksimum segment süresi (None → config: 60.0)

    DÖNER:
      BatchResult — tüm segment sonuçları + özet istatistikler.
    """
    backend = build_backend(VLM_CONFIG)
    if not hasattr(backend, "infer_video"):
        raise RuntimeError(
            "Segment bazlı analiz video destekli bir backend gerektirir "
            "(config.yaml → vlm.default_backend: server)."
        )

    os.makedirs(output_dir, exist_ok=True)
    clips = prepare_video_segments(
        video_path,
        os.path.join(output_dir, "segments"),
        max_segment_sec=max_segment_sec,
        ssim_threshold=ssim_threshold,
        min_segment_sec=min_segment_sec,
    )
    print(f"[pipeline] {len(clips)} segment işlenecek: {video_id}")

    memory = VideoAnalysisMemory()
    segments: list[SegmentResult] = []
    failed: list[dict] = []

    for clip in clips:
        print(
            f"[pipeline] Segment {clip.index + 1}/{len(clips)}: "
            f"{clip.start_sec:.1f}sn – {clip.end_sec:.1f}sn"
        )
        try:
            interpretation = backend.infer_video(
                video_path=clip.path,
                video_id=video_id,
                clip_start_sec=clip.start_sec,
                clip_end_sec=clip.end_sec,
                memory_context=memory.to_prompt_context(),
                is_segment=len(clips) > 1,
            )
            memory.absorb(interpretation, clip.start_sec)
            segments.append(SegmentResult(
                segment_index=clip.index,
                start_sec=clip.start_sec,
                end_sec=clip.end_sec,
                interpretation=interpretation,
            ))
            _write_json(
                interpretation.to_dict(),
                os.path.join(
                    output_dir,
                    f"{interpretation.packet_id}_seg{clip.index:03d}_vlm_interpretation.json",
                ),
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(f"[pipeline] UYARI: Segment {clip.index} işlenemedi ({err}), atlanıyor.")
            failed.append({
                "segment_index": clip.index,
                "start_sec": clip.start_sec,
                "end_sec": clip.end_sec,
                "error": err,
            })

    result = BatchResult(
        video_id=video_id,
        segments=segments,
        failed_segments=failed,
        memory_context=memory.to_prompt_context(),
    )
    _write_json(result.to_dict(), os.path.join(output_dir, f"{video_id}_batch.json"))

    critical = result.most_critical_segment
    print(
        f"[pipeline] Tamamlandı: {result.total_segments} segment, "
        f"{result.total_risk_events} risk olayı, "
        f"en kritik segment: {critical.segment_index if critical else 'yok'}"
    )
    return result


# ---------------------------------------------------------------------------
# Doğrudan çalıştırma — hızlı test için
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    video = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"
    out = run_channel_b(video, video_id=os.path.splitext(os.path.basename(video))[0])
    print(json.dumps(out, ensure_ascii=False, indent=2)[:3000])
