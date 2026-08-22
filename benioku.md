# Benioku — AI Ajanları İçin Kod Tabanı Rehberi

Bu dosya, projede çalışacak AI ajanlarının (ve yeni geliştiricilerin) kodu hızla
anlaması için yazılmıştır. **Proje tanıtımı ve kurulum için `README.md`**
(yarışma odaklı), **açık işler için `YAPILACAKLAR.md`** dosyasına bak.

**Proje:** Dalga AI — TEKNOFEST 2026 Türkçe Yapay Zeka Ajanları, Senaryo 3
(İSG video analiz ve karar destek). Tamamen yerel çalışır; harici API yok.
**Detaylı mimari plan:** `/run/media/bera/çalışma/000_İndeksleme/Projeler/dalgai/plan/` (00–13 numaralı dosyalar; 13 numaralı dosya modül arayüz sözleşmeleridir).

---

## 1. Mimari — İki Kanallı Hat

```
video.mp4
  └─ preprocessing: VideoReader (PyAV) → FrameSampler (8 kare, SSIM+Laplacian)
       → LowLightEnhancer (CLAHE) → resize 384x216
  │
  ├─ KANAL A (hızlı taraf, objektif):
  │    ObserverAgent (detector + tracker + SceneGraph)
  │      → EventEngine (geometrik kurallar, state machine) → event_signals
  │      → select_critical_frames (olay + hareket skorlu 4 kare)
  │      → RAGLayer (risk pattern + aksiyon eşleme) → rag_context
  │
  ├─ KANAL B (VLM tarafı, bağımsız):
  │    DecisionAgent.interpret_frames(kritik kareler, GENERAL_OBSERVATION_PROMPT)
  │      → vlm_observation (GENEL terimler: "araç", "kişi"; spesifik sınıf YOK)
  │
  └─ BİRLEŞİM: DecisionAgent.decide()
       (sinyaller + RAG + scene graph + vlm_observation + kısa süreli hafıza)
       → OutputGuardrail (Pydantic şema + retry + "Bilmiyorum")
       → outputs/analysis_result.json + outputs/metrics.json
```

Temel ilke: **nihai karar hiçbir zaman boşluktan konuşmaz.** Spesifik nesne
tanımları (forklift, palet, baret...) algı katmanından gelir; VLM kendi
gördüklerini genel terimlerle betimler; çelişkiyi Karar Ajanı çözer ve
`reasoning` alanında belirtir.

## 2. Dizin Haritası ve Modül Sözleşmeleri

### `src/preprocessing/`
- `video_reader.py` — `VideoReader`: PyAV ile kare okuma; `iter_frames()` RGB24
  `np.uint8` ndarray döner; `fps`, `total_frames` property'leri.
- `frame_sampler.py` — `FrameSampler.sample(frames, total_frames)` →
  `(sampled_frames, sampled_indices)`. **Önemli:** `sampled_indices` gerçek video
  kare indeksleridir; timestamp hesabı buna dayanır.
- `enhancer.py` — `LowLightEnhancer.enhance(frame)`: LAB üzerinde CLAHE + gamma.
- `critical_frames.py` — `select_critical_frames(frames, sampled_indices,
  event_signals, fps, max_count)` → `(frames, indices)`. Önce olay
  timestamp'lerine en yakın kareler, kalan kontenjan hareket+keskinlik skoruyla.

### `src/perception/`
- `detector.py` — `Detection` (bbox xyxy, `center`, `to_dict()`, opsiyonel
  `polygon`) ve `ObjectDetector` (Ultralytics YOLO; `supports_tracking=True`).
  **`create_detector(config)` factory'si backend seçer** — yeni backend
  eklerken buraya bak.
- `hf_detector.py` — `HFObjectDetector` (transformers
  `AutoModelForObjectDetection`; şu an `PaddlePaddle/PP-DocLayoutV3_safetensors`).
  `supports_tracking=False` → `model.track()` YOK; takip IoU ile yapılır.
  **Geçicidir; İSG sahnelerinde anlamlı tespit üretmez** — eğitilmiş YOLO
  gelince `detector_backend: "ultralytics"`e dönülecek.
- `tracker.py` — `ObjectTracker.track()` Ultralytics `model.track()` çağırır
  (ByteTrack). `TrackedObject`: `history`, `speed`, `is_stationary`,
  `disappeared` (5 kare tolerans).
- `observer_agent.py` — `ObserverAgent.observe_video(frames, fps,
  sampled_indices)` → kare başına observation dict:
  `{frame_idx, timestamp, detections[], tracks[], scene_graph{}}`.
  **Timestamp = `sampled_indices[idx] / fps`** (örneklem indeksi değil!).
  Detector `supports_tracking=False` ise `_track_by_iou()` (min IoU 0.3) ile
  track ID korunur.
- `scene_graph.py` — `SceneGraph`: düğümler (nesneler), kenarlar (`near` ≤100px,
  `carrying` forklift↔palet ≤150px, `wearing` insan↔baret/yelek).

### `src/events/`
- `event_engine.py` — `EventEngine.process_observation(obs)`; observation
  dict'lerini tekrar `TrackedObject`'e çevirir; aynı track+event_type 10 sn
  içinde tekrarlanmaz. `get_signals()` → `EventSignal.to_dict()` listesi;
  **timestamp alanı "MM:SS" string**'dir (float değil).
- `rules.py` — `RuleSet`: config'teki `enabled_rules`'a göre geometrik kurallar
  (tip_over: en/boy oranı; fall: dikey düşüş; gathering; immobility;
  ppe_missing; proximity; fire_smoke/leakage placeholder).
- `state_machine.py` — `TrackStateMachine`: track geçmişi ve süreklilik durumu.

### `src/reasoning/`
- `rag_layer.py` — `RAGLayer.build_context(event_signals)` →
  `{risk_level, risk_score, actions[], matched_patterns[]}`. Veri:
  `data/risk_patterns.yaml` + `data/action_catalog.yaml`.
- `memory.py` — `ShortTermMemory` (max 100 kayıt sliding window);
  `to_prompt_context()` son 10 olayı metin olarak verir.
- `mock_tools.py` — `MockToolRegistry`: `data/mock_tools.yaml`'dan 5 araç.
  **Bilinen boşluk:** `execute()` ana akışta çağrılmıyor (YAPILACAKLAR §6).
- `decision_agent.py` — iki kritik metod:
  - `interpret_frames(images)` — **Kanal B**: `GENERAL_OBSERVATION_PROMPT` ile
    bağımsız, genel-terimli kare yorumu üretir. Prompt'a spesifik sınıf adı
    BİLİNÇLİ olarak konmaz (ekip kararı: genel tanım = daha yüksek doğruluk).
  - `decide(images, event_signals, scene_graphs, fps, vlm_observation)` —
    `_build_prompt()` bölümleri: sistem promptu → ALGI SİNYALLERİ → RAG →
    SAHNE GRAFİ → VLM KARE YORUMU (GENEL) → HAFIZA → katı JSON şema talimatı.

### `src/models/vlm_backend.py`
`VLMBackend` ABC (`generate(images, prompt, temperature, max_tokens)`);
üç implementasyon: `VLLMBackend` (Qwen2.5-VL), `LlamaCppBackend` (TimeLens-7B
GGUF, Vulkan), `TransformersBackend` (LLaVA-NeXT-Video). `create_backend()`
`auto_preference` sırasıyla dener, biri açılmazsa sıradakine düşer.
**Görüntüler numpy RGB ndarray olarak gider; JPEG base64'e backend çevirir.**

### `src/output/`
- `schema.py` — `AnalysisOutput` (Pydantic): `summary`, `events[]`
  (`time` "MM:SS" pattern'li), `risk` ∈ {Düşük, Orta, Yüksek}, `actions[]`,
  `reasoning`, `confidence`, `triggered_mock_tools[]`.
- `guardrail.py` — `OutputGuardrail.validate(raw, generate_fn, rag_risk_level)`:
  JSON çıkar (```json bloğu veya ilk { ... son }), şemaya zorla, semantic check
  (Yüksek risk → olay + ≥2 aksiyon şart), temperature düşürerek max 3 retry,
  hepsi başarısızsa "Bilmiyorum" null-response.

### `src/main.py` — orkestrasyon
Adım sırası dosyada numaralı yorum bloklarıyla işaretli (1: video oku …
10: kaydet). CLI: `--video --config --backend --output --no-enhance
--save-grid`. Ağır importlar `main()` içinde lazy.

### `src/config.py` + `config.yaml`
Tüm eşikler/model yolları `config.yaml`'de; Pydantic modelleri `config.py`'de.
**Yeni ayar eklerken ikisini birden güncelle.** Önemli anahtarlar:
`preprocessing.critical_frame_count` (4), `perception.detector_backend`
(`hf_transformers` | `ultralytics`), `perception.hf_model`,
`decision_agent.system_prompt`, `vlm.auto_preference`.

## 3. Konvansiyonlar ve Kurallar

- **Dil:** kod yorumları ve dokümantasyon Türkçe; tanımlayıcılar İngilizce.
- **Tip kullanımı:** `from __future__ import annotations`, `NDArray[np.uint8]`
  kare tipi (RGB, HWC).
- **Lazy import:** ağır bağımlılıklar (torch, transformers, ultralytics, vllm)
  fonksiyon/`_load()` içinde import edilir — model yüklemeyen testler ve
  `--help` bağımlılıksız çalışır. Bu deseni bozma.
- **Sınıf eşleme:** COCO → Türkçe (`person→insan`, `truck/car→forklift`);
  `custom_classes` listesi varsa listede olmayan sınıflar eşlenmeden geçer.
- **Eşikler:** config'teki sayısal değerler başlangıç önerisidir; modül sahibi
  kendi testleriyle kesinleştirir. Sabit olan tek şey modül arayüzleri.

## 4. Çalıştırma ve Test

```bash
python -m src.main --video video.mp4 --save-grid      # tam pipeline
python -m src.main --backend llama_cpp                 # backend zorla
python -m pytest tests/ -v                             # 18 test
```

Test notu: proje diski (`/run/media/...`, NTFS) symlink desteklemediği için
venv `/tmp/bera_test_venv`'de kuruludur; testler onunla koşulur:
`/tmp/bera_test_venv/bin/python -m pytest tests/ -v`.
HF detector/VLM ile uçtan uca koşu model indirme + GPU ister; CI/testlerde
model yükleme mock'lanır.

## 5. Bilinen Tuzaklar

1. **Timestamp:** observation timestamp'i `sampled_indices` üzerinden gelir;
   `frame_idx` örneklem pozisyonudur, video karesi değil — karıştırma.
2. **EventSignal timestamp "MM:SS" string**; sayıya çevirirken
   `critical_frames._timestamp_to_seconds` veya `main._time_to_seconds` kullan.
3. **HF backend + tracker:** `HFObjectDetector`'da `.track()` yok;
   `ObjectTracker.track()` sadece Ultralytics ile çalışır.
4. **Guardrail retry** `agent._build_prompt()`'u doğrudan çağırır — imzasını
   değiştirirsen `main.py`'deki closure'ı da güncelle.
5. **PP-DocLayoutV3 geçicidir** — üretime dair tespit bekleme; gerçek algı
   eğitilecek İSG YOLO'su ile gelecek (bkz. `YAPILACAKLAR.md` §1–2).

---

## 6. UI Katmanı (Yeni)

### Ekranlar
- **Süpervizör (`/`)** — 9'lu pseudo-live kamera grid; riskli olaylarda kırmızı
  yanıp sönen çerçeve; kamera tıklayınca tam ekran modal; sağ panelde
  bildirimler ve aksiyon butonları. "Saha Ekibine Bildirim Gönder" gerçek
  sistem-içi push'tur, diğer aksiyonlar mock gösterimdir.
- **Saha Ekibi (`/saha.html`)** — rol filtreli alarm listesi; riskli video
  kesiti otomatik oynatılır; yeni alarmda kırmızı banner + bip sesi.
- **Admin (`/admin.html`)** — KPI'lar, olay tablosu, post-it görünümlü
  İSG çalışma alanı önerileri; öneriye tıklayınca LLM chat drawer açılır.
  Öneriler admin tarafından elle tetiklenir (`/api/v1/suggestions/query`).

### Çalıştırma
```bash
uvicorn backend.gateway.main:app --host 0.0.0.0 --port 8000
```
Tarayıcıda: `http://localhost:8000/`, `http://localhost:8000/saha.html`,
`http://localhost:8000/admin.html`.

### Pseudo-live veri kaynağı
- Placeholder analizler: `data/pseudolive/analyses/analysis_01.json ... analysis_27.json`
- Placeholder videolar: `data/pseudolive/videos/video_01.mp4 ... video_27.mp4`
- Gerçek verilerle değiştirmek için aynı dosya adlarını koruyun.

### Önemli Not
UI, **normal AI risk tespit akışına dahil değildir**. `isg_onerileri.yaml`
yeni bir RAG koleksiyonudur ve sadece admin paneli öneri/chat akışında
kullanılır; `RAGLayer.build_context()` ve karar ajanı aynı kalmıştır.
