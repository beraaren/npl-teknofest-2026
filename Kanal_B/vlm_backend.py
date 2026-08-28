
from __future__ import annotations
import base64
import json
import os
import re
import time
from abc import ABC, abstractmethod

import requests

# S1b ve S8 sözleşme yapıları
from contracts import (
    VLMInterpretation, DetectedEntity, InferenceMetrics, VLMFramePacket,
    RiskEvent, format_mmss,
)


# ---------------------------------------------------------------------------
# Sistem Promptu — model davranışını yönlendiren temel talimat
# ---------------------------------------------------------------------------

def build_system_prompt(rows: int, cols: int) -> str:
    """Grid boyutuna göre dinamik sistem promptu üretir.

    NEDEN DİNAMİK?
      Önceki SYSTEM_PROMPT sabit "2×4 grid için 0-7" içeriyordu.
      Ancak grid boyutu config.yaml'dan değişebilir (ör. 3×3 = 9 hücre).
      Bu fonksiyon rows/cols'u packet.grid_layout'tan alarak doğru
      hücre sayısını ve indeks aralığını prompt'a enjekte eder.
    """
    total_cells = rows * cols
    max_idx = total_cells - 1
    return (
        "You are a security camera scene analyst. You are given a single grid image "
        "containing frames taken from a video stream. "
        f"The grid is {rows}×{cols}, cells are numbered left-to-right, top-to-bottom "
        f"starting from 0 (cells 0-{max_idx}, {total_cells} total). "
        "Your task is to interpret the scene with your own eyes, INDEPENDENTLY of the "
        "event engine and the object detector.\n\n"
        "VEHICLE IDENTIFICATION — THINK STEP BY STEP:\n"
        "When you see ANY vehicle or machine, reason about its exact type before labeling it. "
        "Consider visual cues: size, shape, wheels vs tracks, cabin position, forks, boom arm, "
        "bucket, flatbed, road context vs industrial site. Possible types include but are not "
        "limited to: forklift, crane, excavator, loader, truck, pickup, car, van, bus, "
        "motorcycle, bicycle. A vehicle on a public road is most likely a car/truck/bus — "
        "do NOT assume it is industrial equipment. Write your reasoning in the "
        "\"vehicle_type_reasoning\" field.\n\n"
        "Produce an output that matches ONLY the following JSON schema and "
        "contains no other text:\n"
        "{\n"
        '  "scene_summary_tr": "1-3 sentence scene summary, in English",\n'
        '  "vehicle_type_reasoning": "Step-by-step reasoning about what type each vehicle/machine is and why",\n'
        '  "detected_entities": [{"label": "specific vehicle type or object label", "confidence_hint": "low|medium|high", "notes_tr": "short note, in English"}],\n'
        '  "detected_actions_tr": ["observed actions, in English"],\n'
        '  "risk_events": [\n'
        '    {\n'
        '      "description_tr": "risk description, in English",\n'
        '      "severity": "low|medium|high|critical",\n'
        '      "confidence": 0.85,\n'
        '      "supporting_frame_count": 2,\n'
        '      "supporting_frame_positions": [1, 4]\n'
        '    }\n'
        '  ],\n'
        '  "confidence_overall": 0.0\n'
        "}\n"
        "risk_events RULES: return an empty list [] if there is no risk. "
        "severity: low=low priority, medium=attention, high=significant risk, critical=human health in danger. "
        f"supporting_frame_count: in how many grid cells you saw this risk (0-{total_cells}). "
        f"supporting_frame_positions: grid indices where the risk is visible (0=top-left, left-to-right, max={max_idx}). "
        "confidence: your risk-specific confidence (0.0-1.0)."
        # NEDEN YENİ ŞEMA?
        # Önceki risk_flags_tr (düz string listesi) hangi riskin ne kadar kritik
        # olduğunu ve kaç karede görüldüğünü Karar Ajanı'na söylemiyordu.
        # risk_events ile her risk artık severity + kanıt bilgisiyle taşınır.
        # Dil İngilizce: VLM'ler İngilizce talimatla daha tutarlı JSON üretiyor
        # (main dalındaki "ingilizceye çevrildi" değişikliği korundu).
    )


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def _encode_jpeg_b64(image_path: str) -> str:
    """Grid JPEG görselini base64 string'e dönüştürür.

    Neden base64?
      OpenAI-uyumlu API'ler görsel veriyi data URI formatında bekler:
      "data:image/jpeg;base64,{b64}" → HTTP request body'sine gömülür.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_json(text: str) -> dict:
    """VLM'in ham metin çıktısından JSON bloğunu çıkarır ve parse eder.

    NOT: Bu fonksiyon <think>...</think> bloğu çıkarıldıktan sonraki metni alır.
    Bkz. _extract_reasoning_and_json() — reasoning parse için üst düzey fonksiyon.

    3 kademeli strateji — Model farklı formatlarda yanıt üretebilir:

    1. Kademe — Direkt parse (en hızlı):
       Model tam ve düzgün JSON ürettiyse doğrudan json.loads() çalışır.
       Örnek: {"scene_summary_tr": "...", ...}

    2. Kademe — Markdown fence arama:
       Model açıklama + JSON bloğu ürettiyse fence'i yakala.
       Örnek: "Tabii, işte analiz:\n```json\n{...}\n```"

    3. Kademe — Greedy { ... } arama (eski davranış, son çare):
       Model hem metin hem JSON ürettiyse ilk JSON bloğunu bul.
       Örnek: "Sahne analizi şu şekilde: {...}"

    Tüm kademeler başarısız olursa ValueError fırlatır → infer() fallback'e düşer.
    """
    stripped = text.strip()

    # 1. Kademe: direkt parse
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 2. Kademe: markdown fence içinde ara (```json ... ``` veya ``` ... ```)
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Kademe: ilk { ... } bloğunu greedy yakala
    brace = re.search(r"\{.*\}", stripped, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"VLM çıktısında geçerli JSON bulunamadı: {text[:200]}")


def _extract_reasoning_and_json(text: str) -> tuple[str, dict]:
    """Ham VLM çıktısından reasoning trace ve JSON'ı ayrı ayrı çıkarır.

    [REASONING] Qwen2.5-VL ve benzeri thinking-capable modeller yanıtlarını
    <think>...</think> bloğuyla başlatır. Bu blok:
      - Karar Ajanı için ek bağlam sağlar (yüksek confidence + güçlü reasoning)
      - Debug/audit trail için reasoning_trace alanına kaydedilir
      - JSON parse'ı etkilememesi için önceden çıkarılır

    Model reasoning üretmediyse reasoning="" (boş string) döner.
    Kalan metinden 3 kademeli JSON parse uygulanır (_extract_json).

    Döner: (reasoning_trace_str, parsed_dict)
    """
    reasoning = ""
    # <think>...</think> veya <düşünme>...</düşünme> bloğunu (non-greedy) ayıkla.
    # İki etiket desteklenir:
    #   <think>     → Qwen2.5-VL ve benzeri modellerin standart thinking etiketi
    #   <düşünme>   → eski Türkçe prompt şablonunun etiketi (geriye dönük uyumluluk)
    # Her iki etiket de reasoning_trace'e kaydedilir; JSON parse'ı etkilemez.
    think_match = re.search(
        r"<(?:think|düşünme)>(.*?)</(?:think|düşünme)>",
        text,
        re.DOTALL,
    )
    if think_match:
        reasoning = think_match.group(1).strip()
        # Bloğu metinden çıkar → kalan metin JSON parse için hazır
        text = text[: think_match.start()] + text[think_match.end() :]
    return reasoning, _extract_json(text.strip())


def _compute_confidence(parsed: dict, parse_succeeded: bool) -> float:
    """VLM çıktısından BAĞIMSIZ güven skoru hesaplar (Değişiklik 1).

    Model öz güveninden (confidence_overall) FARKLI olarak, bu skor dışsal
    sinyallere dayanır ve Karar Ajanı'na daha güvenilir bir kalibrasyon sunar:
      - Parse başarısı   : başarısızsa anında 0.0 döner.
      - Risk tutarlılığı: yüksek severity + yüksek confidence → skor artar.
      - Kanıt çeşitliliği: ortalama supporting_frame_count / 8 → max 0.2 bonus.
      - Model öz güveni : %50 ağırlıkla bileşime girer.

    Ağırlık formülü: model_conf×0.5 + risk_score×0.3 + evidence_bonus (max 0.2)
    """
    if not parse_succeeded:
        return 0.0

    model_conf = float(parsed.get("confidence_overall", 0.0))

    raw_risks = [r for r in parsed.get("risk_events", []) if isinstance(r, dict)]
    if raw_risks:
        _severity_w = {"low": 0.25, "medium": 0.50, "high": 0.75, "critical": 1.00}
        weighted = [
            float(r.get("confidence", 0.5))
            * _severity_w.get(r.get("severity", "low"), 0.25)
            for r in raw_risks
        ]
        risk_score = min(sum(weighted) / max(len(weighted), 1), 1.0)
        avg_frames = (
            sum(int(r.get("supporting_frame_count", 1)) for r in raw_risks)
            / len(raw_risks)
        )
        evidence_bonus = min(avg_frames / 8.0, 1.0) * 0.2
    else:
        risk_score     = 0.0
        evidence_bonus = 0.0

    computed = (model_conf * 0.5) + (risk_score * 0.3) + evidence_bonus
    return round(min(1.0, max(0.0, computed)), 4)


# ---------------------------------------------------------------------------
# Abstract base class — ortak inference arayüzü
# ---------------------------------------------------------------------------

class VLMBackendBase(ABC):
    """Tüm VLM backend'lerinin uyguladığı soyut arayüz.

    TASARIM KARARI: Adapter Pattern
      Her backend _raw_infer() metodunu uygular (raw metin + token sayısı döner).
      infer() metodu ortak: latency ölçer, JSON parse eder, S8 paketi oluşturur.
      Karar Ajanı sadece infer(packet) → VLMInterpretation döngüsünü bilir.

    Yeni bir backend eklemek için:
      1. Bu sınıftan türet
      2. backend_name sınıf değişkenini ayarla
      3. _raw_infer() metodunu uygula
      4. build_backend() factory'e ekle
    """
    backend_name: str   # "vllm" | "llama.cpp" | "transformers"
    model_name: str     # kullanılan modelin HF hub adı veya dosya adı

    @abstractmethod
    def _raw_infer(self, image_path: str, prompt: str) -> tuple[str, int]:
        """Ham VLM çağrısı — her backend bunu farklı uygular.
        Döner: (ham_metin, üretilen_token_sayısı)
        """

    def infer(self, packet: VLMFramePacket) -> VLMInterpretation:
        """S1b paketini alır, VLM'e gönderir, S8 paketi döndürür.

        Bu metod tüm backend'lerde ortaktır:
          1. Grid boyutuna göre dinamik sistem promptu oluştur (build_system_prompt)
          2. _raw_infer() çağrısını zamanla (latency_ms hesabı için)
          3. <think>...</think> reasoning trace'i ayıkla, JSON'ı parse et
          4. Parse başarısız → S8 sözleşmesini bozmadan fallback dict kullan
          5. RiskEvent listesi oluştur (Değişiklik 5 & 6)
          6. Bağımsız güven skorunu hesapla (Değişiklik 1)
          7. VLMInterpretation (S8) oluştur ve döndür (reasoning_trace dahil)
        """
        # Adım 1: Grid boyutuna göre dinamik prompt — sabit "0-7" yerine doğru aralık
        system_prompt = build_system_prompt(
            packet.grid_layout.rows, packet.grid_layout.cols
        )

        t0 = time.perf_counter()
        raw_text, tokens = self._raw_infer(packet.grid_image_path, system_prompt)
        latency_ms = (time.perf_counter() - t0) * 1000  # sn -> ms

        parse_succeeded = True
        reasoning_trace = ""
        try:
            # [REASONING] <think>...</think> bloğunu ayıkla, kalan metni parse et
            reasoning_trace, parsed = _extract_reasoning_and_json(raw_text)
        except ValueError:
            parse_succeeded = False
            # Model şemaya uymayan çıktı verdiyse S8 sözleşmesini bozmadan devam et.
            parsed = {
                "scene_summary_tr": "Model çıktısı ayrıştırılamadı.",
                "detected_entities": [],
                "detected_actions_tr": [],
                "risk_events": [],
                "confidence_overall": 0.0,
            }

        # detected_entities listesini DetectedEntity dataclass'larına dönüştür
        entities = [
            DetectedEntity(
                label=e.get("label", ""),
                confidence_hint=e.get("confidence_hint", "low"),
                notes_tr=e.get("notes_tr", ""),
            )
            for e in parsed.get("detected_entities", [])
            if isinstance(e, dict)
        ]

        # [DEĞİŞİKLİK 5 & 6] risk_events → RiskEvent listesi
        risk_events = [
            RiskEvent(
                description_tr=r.get("description_tr", ""),
                severity=r.get("severity", "low"),
                confidence=float(r.get("confidence", 0.5)),
                supporting_frame_count=int(r.get("supporting_frame_count", 1)),
                supporting_frame_positions=r.get("supporting_frame_positions", []),
            )
            for r in parsed.get("risk_events", [])
            if isinstance(r, dict)
        ]

        # [DEĞİŞİKLİK 1] Bağımsız güven skoru
        computed_conf = _compute_confidence(parsed, parse_succeeded)

        # S8 sözleşme paketi — Karar Ajanı'nın 3. girdisi
        return VLMInterpretation(
            packet_id=packet.packet_id,       # S1b ile izlenebilirlik bağlantısı
            video_id=packet.video_id,
            model_name=self.model_name,
            model_backend=self.backend_name,
            scene_summary_tr=parsed.get("scene_summary_tr", ""),
            detected_entities=entities,
            detected_actions_tr=parsed.get("detected_actions_tr", []),
            risk_events=risk_events,
            confidence_overall=float(parsed.get("confidence_overall", 0.0)),
            computed_confidence=computed_conf,
            inference=InferenceMetrics(latency_ms=latency_ms, tokens_generated=tokens),
            raw_model_output=raw_text,        # debug/audit için ham çıktı
            reasoning_trace=reasoning_trace,  # [REASONING] <think>...</think> içeriği
        )


# ---------------------------------------------------------------------------
# Backend 1 — vLLM (GPU hızlı)
# ---------------------------------------------------------------------------

class VLLMBackend(VLMBackendBase):
    """vLLM OpenAI-uyumlu sunucu ile inference.

    KULLANIM KOŞULU:
      vLLM sunucusu görsel destekli bir model ile ayağa kaldırılmış olmalı.
      Benimki: vllm serve Qwen/Qwen2.5-VL-7B-Instruct --tensor-parallel-size 1

    MESAJ YAPISI (S6 değişikliği):
      system: SYSTEM_PROMPT (talimatlar)
      user  : grid görseli (base64 data URI)

         Model system rolünü desteklemiyorsa system bloğunu kaldır,
         prompt'u user content'inin başına {"type":"text","text":prompt} olarak ekle.
    """
    backend_name = "vllm"

    def __init__(self, base_url: str, model_name: str,
                 max_tokens: int = 512, temperature: float = 0.2,
                 timeout_sec: int = 60):
        # S5: temperature artık __init__ parametresi → config.yaml'dan gelir
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature  # config.yaml'da vllm.temperature: 0.15
        self.timeout_sec = timeout_sec  # [#5] HTTP timeout → config.yaml'dan

    def _raw_infer(self, image_path: str, prompt: str) -> tuple[str, int]:
        b64 = _encode_jpeg_b64(image_path)
        payload = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,  # S5: hardcode 0.2
            # S6: SYSTEM_PROMPT ayrı "system" rolünde → user sadece görsel içerir
            "messages": [
                {
                    "role": "system",
                    "content": prompt,        # SYSTEM_PROMPT burada
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                },
            ],
        }
        resp = requests.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()  # HTTP hatalarını yukarı taşı
        data = resp.json()
        text   = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return text, tokens


# ---------------------------------------------------------------------------
# Backend 2 — llama.cpp (GGUF model, CPU veya küçük GPU için)
# ---------------------------------------------------------------------------

class LlamaCppBackend(VLMBackendBase):
    """llama.cpp'nin llama-server'ı ile inference.

    KULLANIM KOŞULU:
      llama-server multimodal GGUF model + mmproj dosyasıyla çalışıyor olmalı.
      Örnek: llama-server -m model.gguf --mmproj mmproj.gguf -c 16384

    ! LLaVA tabanlı bazı GGUF modeller system rolünü desteklemez.
       Desteklemiyorsa bu sınıftaki messages'tan system bloğunu kaldır ve
       prompt'u user content'e text olarak ekle:
       {"type": "text", "text": prompt}
    """
    backend_name = "llama.cpp"

    def __init__(self, base_url: str, model_name: str,
                 max_tokens: int = 512, temperature: float = 0.2,
                 timeout_sec: int = 60):
        # S5: temperature artık __init__ parametresi → config.yaml'dan gelir
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature  # config.yaml'da llama_cpp.temperature: 0.15
        self.timeout_sec = timeout_sec  # [#5] HTTP timeout → config.yaml'dan

    def _raw_infer(self, image_path: str, prompt: str) -> tuple[str, int]:
        b64 = _encode_jpeg_b64(image_path)
        payload = {
            "model":self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,  # S5: hardcode 0.2 → self.temperature
            # S6: SYSTEM_PROMPT ayrı "system" rolünde
            "messages": [
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                },
            ],
        }
        resp = requests.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        data   = resp.json()
        text   = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        return text, tokens


# ---------------------------------------------------------------------------
# Backend 3 — HuggingFace Transformers (sunucusuz, geliştirme/test için)
# ---------------------------------------------------------------------------

class TransformersBackend(VLMBackendBase):
    """HuggingFace transformers ile doğrudan model yüklemesi.

    KULLANIM KOŞULU:
      GPU varsa device="cuda", yoksa device="cpu" kullan (yavaş).
      Üretimde vLLM veya llama.cpp önerilir (daha hızlı, daha az bellek).

    LAZY LOADING:
      Model ilk _raw_infer() çağrısında yüklenir (ağır modeli hemen yükleme).
      Bu sayede backend nesnesi oluşturulduğunda bellek kullanılmaz.
    """
    backend_name = "transformers"

    def __init__(self, model_name: str, device: str = "cuda", max_new_tokens: int = 512):
        self.model_name      = model_name
        self.max_new_tokens  = max_new_tokens
        self._model          = None       # lazy load: ilk çağrıda yüklenir
        self._processor      = None       # lazy load: ilk çağrıda yüklenir
        self._device         = device

    def _lazy_load(self):
        """Modeli ilk çağrıda yükle (lazy initialization pattern).

        Neden lazy?
          TransformersBackend nesnesi oluşturulduğunda model yüklenmez.
          İlk infer() çağrısında yüklenir → başlangıç süresi optimize edilir.
        """
        if self._model is not None:
            return  # zaten yüklü, tekrar yükleme
        from transformers import AutoProcessor, AutoModelForImageTextToText
        import torch
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model     = AutoModelForImageTextToText.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map=self._device
        )

    def _raw_infer(self, image_path: str, prompt: str) -> tuple[str, int]:
        self._lazy_load()
        from PIL import Image

        image = Image.open(image_path).convert("RGB")

        # Transformers API: görsel ve metin birlikte user mesajına girer
        # (system rol desteği modele göre değişir, burada user'a ekliyoruz)
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        }]
        chat_prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs      = self._processor(text=chat_prompt, images=image, return_tensors="pt").to(self._device)
        output      = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        # Sadece yeni üretilen token'ları decode et (giriş token'larını çıkar)
        text             = self._processor.decode(output[0], skip_special_tokens=True)
        tokens_generated = output.shape[-1] - inputs["input_ids"].shape[-1]
        return text, int(tokens_generated)


# ---------------------------------------------------------------------------
# Factory fonksiyonu — pipeline.py bu fonksiyonu çağırır
# ---------------------------------------------------------------------------

def build_backend(config: dict) -> VLMBackendBase:
    """Config dict'inden uygun backend nesnesini oluşturur.

    config dict'i pipeline.py'nin _load_vlm_config() fonksiyonundan gelir,
    o da config.yaml'daki [vlm:] bloğundan okunur.

    config['backend'] beklenen değerler: 'vllm' | 'llama.cpp' | 'transformers'

    S5 değişikliği:
      temperature artık config.get("temperature", 0.15) ile geçiriliyor.
      config.yaml'da vllm.temperature: 0.15 ve llama_cpp.temperature: 0.15 tanımlı.
    """
    backend = config["backend"]

    if backend == "server":
        # TEKNOFEST EVREN (veya herhangi bir OpenAI-uyumlu servis).
        # Anahtar yapılandırmaya yazılmaz; ortam değişkeninden okunur.
        api_key = os.environ.get(config.get("api_key_env", "EVREN_API_KEY"), "").strip()
        if not api_key:
            print(
                f"[backend] UYARI: '{config.get('api_key_env', 'EVREN_API_KEY')}' "
                f"ortam değişkeni boş; istek kimlik doğrulaması olmadan gidecek."
            )
        return ServerBackend(
            base_url=config["base_url"],
            model_name=config["model_name"],
            video_model=config.get("video_model"),
            api_key=api_key or None,
            max_tokens=config.get("max_tokens", 65536),
            temperature=config.get("temperature", 0.15),
            timeout_sec=config.get("timeout_sec", 1800),
            enable_thinking=config.get("enable_thinking", False),
        )

    if backend == "vllm":
        # S5: temperature config'den geçiyor (config.yaml: vllm.temperature: 0.15)
        # [#5]: timeout_sec config'den geçiyor (config.yaml: vllm.timeout_sec: 120)
        return VLLMBackend(
            config["base_url"], config["model_name"],
            config.get("max_tokens", 512), config.get("temperature", 0.15),
            config.get("timeout_sec", 60),
        )

    if backend == "llama.cpp":
        return LlamaCppBackend(
            config["base_url"], config["model_name"],
            config.get("max_tokens", 512), config.get("temperature", 0.15),
            config.get("timeout_sec", 60),
        )

    if backend == "transformers":
        return TransformersBackend(
            config["model_name"],
            config.get("device", "cuda"),
            config.get("max_tokens", 512),
        )

    raise ValueError(
        f"Bilinmeyen backend: {backend!r} — 'server', 'vllm', 'llama.cpp' "
        f"veya 'transformers' olmalı"
    )


# ===========================================================================
# EVREN SERVER BACKEND — video destekli, OpenAI-uyumlu harici çıkarım servisi
# ===========================================================================

# EVREN gövde sınırı 256 MB; base64 kodlaması boyutu ~1/3 artırdığı için ham
# dosya sınırı pratikte ~190 MB'a karşılık gelir.
_RAW_VIDEO_LIMIT_BYTES = int(256 * 1024 * 1024 / 1.34)


def build_video_system_prompt(
    clip_start_sec: float,
    clip_end_sec: float,
    memory_context: str = "",
    is_segment: bool = False,
) -> str:
    """Video modu için S8 şemasını isteyen prompt'u üretir.

    Üç şeyi açıkça bildirmek zorundayız:

    1. **Mutlak zaman ekseni.** Klip uzun bir kaydın parçası olabilir. Model
       klibin kendi ``0:00``ını videonun başı sanırsa tüm zaman damgaları
       kayar. Ölçümde model 10 saniyelik bir klipte ``00:00:55:08`` gibi
       tutarsız damgalar üretebildiği için zaman, saniye cinsinden **sayı**
       olarak istenir (metin biçim yerine) ve aralık açıkça verilir.
    2. **Önceki segmentlerin bağlamı.** Metin hafıza (bkz.
       :class:`contracts.VideoAnalysisMemory`) prompt'a eklenir; böylece süregelen
       durumlar tekrar "yeni olay" sayılmaz.
    3. **Genel terim kullanımı.** Spesifik sınıf adları (forklift, palet)
       bilinçli olarak istenmez; onları algı katmanı (YOLO) sağlar. Kanal B'nin
       değeri bağımsız ve önyargısız görsel yorum olmasıdır.

    Args:
        clip_start_sec: Klibin tam videodaki mutlak başlangıcı.
        clip_end_sec: Klibin tam videodaki mutlak bitişi.
        memory_context: Önceki segmentlerden taşınan metin hafıza (boş olabilir).
        is_segment: Klip, daha uzun bir videonun parçası mı.

    Returns:
        Modele gönderilecek tam prompt metni.
    """
    start_label = format_mmss(clip_start_sec)
    end_label = format_mmss(clip_end_sec)

    if is_segment:
        time_rules = (
            f"IMPORTANT — ABSOLUTE TIME AXIS: This clip is a SEGMENT of a longer "
            f"recording. The clip's own 00:00 corresponds to {start_label} "
            f"({clip_start_sec:.1f} seconds) of the FULL video. The segment covers "
            f"{start_label}-{end_label} of the full video.\n"
            f"Every timestamp you report in \"timestamp_sec\" MUST be in ABSOLUTE "
            f"full-video seconds, i.e. a number between {clip_start_sec:.1f} and "
            f"{clip_end_sec:.1f}. Do NOT report time relative to the clip start.\n\n"
        )
    else:
        time_rules = (
            f"The clip covers the whole video, {start_label}-{end_label} "
            f"(0.0 to {clip_end_sec:.1f} seconds).\n"
            f"Report \"timestamp_sec\" as a number of seconds between 0.0 and "
            f"{clip_end_sec:.1f}.\n\n"
        )

    memory_block = f"{memory_context}\n\n" if memory_context.strip() else ""

    return (
        "You are an occupational health and safety (OHS) video analyst. You are "
        "given a video clip from a work site or surveillance camera. Interpret the "
        "scene with your own eyes, INDEPENDENTLY of any object detector or rule "
        "engine.\n\n"
        + time_rules
        + memory_block
        + "OBJECT NAMING: Use GENERAL terms for what you see: 'person', 'vehicle', "
        "'machine', 'load', 'rack/structure', 'liquid/substance', 'smoke or flame'. "
        "Do not guess specific equipment model names. If a vehicle's type is clearly "
        "visible you may name it, but state your uncertainty in notes_tr.\n\n"
        "Produce output matching ONLY the following JSON schema, with no other text:\n"
        "{\n"
        '  "scene_summary_tr": "1-3 sentence summary of the scene, in Turkish",\n'
        '  "detected_entities": [{"label": "general object label", '
        '"confidence_hint": "low|medium|high", "notes_tr": "short note in Turkish"}],\n'
        '  "detected_actions_tr": ["observed actions, in Turkish"],\n'
        '  "risk_events": [\n'
        '    {\n'
        '      "description_tr": "risk description, in Turkish",\n'
        '      "severity": "low|medium|high|critical",\n'
        '      "confidence": 0.85,\n'
        f'      "timestamp_sec": {clip_start_sec:.1f}\n'
        '    }\n'
        '  ],\n'
        '  "confidence_overall": 0.0\n'
        "}\n"
        "risk_events RULES: return an empty list [] if there is no risk. "
        "severity: low=low priority, medium=needs attention, high=significant risk, "
        "critical=human health in immediate danger. "
        "confidence: your risk-specific confidence (0.0-1.0). "
        "timestamp_sec: the absolute second in the full video when the risk is visible."
    )


class ServerBackend(VLMBackendBase):
    """OpenAI-uyumlu harici çıkarım servisi (TEKNOFEST EVREN) — video + grid.

    Bu backend iki modu da destekler:

    * :meth:`infer` (tabandan gelir) — grid JPEG'i görüntü olarak gönderir.
      EVREN'de görüntü kabul eden modeller ``llm-fast``/``llm-large``dır.
    * :meth:`infer_video` — videoyu doğrudan gönderir. EVREN'de video
      analizine özelleşmiş alias ``vlm``dir ve **görüntü kabul etmez**; bu
      yüzden iki mod için iki ayrı model adı tutulur.

    Ön ek önbelleği (prefix cache): video baytları birebir aynı kaldığı sürece
    aynı klip üzerinden sorulan takip soruları önbellekten yararlanır. Bu
    nedenle base64 kodlaması :meth:`encode_video` ile bir kez yapılıp
    :meth:`infer_video`'ya ``video_b64`` olarak geçirilebilir.
    """

    backend_name = "server"

    def __init__(
        self,
        base_url: str,
        model_name: str,
        video_model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 65536,
        temperature: float = 0.15,
        timeout_sec: int = 1800,
        enable_thinking: bool | None = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name          # görüntü/metin modeli
        self.video_model = video_model or model_name
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_sec = timeout_sec
        # Akıl yürütme kapalı: açıkken üretim token'ı dalgalanıp bütçeyi taşırıyor
        # ve ayrıştırıcı izi sildiği için yanıt HTTP 200 ile boş dönebiliyor.
        self.enable_thinking = enable_thinking

    # -- HTTP ------------------------------------------------------------

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, model: str, content: list, max_tokens: int | None = None) -> tuple[str, int]:
        """``/chat/completions`` çağrısı yapar; (metin, token) döner."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "temperature": self.temperature,
        }
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(), json=payload, timeout=self.timeout_sec,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Çıkarım servisi HTTP {resp.status_code}: {resp.text[:400]}"
            )
        data = resp.json()
        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        tokens = int(data.get("usage", {}).get("completion_tokens", 0))

        if not text.strip():
            raise RuntimeError(
                f"Model boş içerik döndürdü (finish_reason="
                f"{choice.get('finish_reason')!r}, max_tokens={payload['max_tokens']}). "
                f"Akıl yürütme açıksa token bütçesi izi tarafından tükenmiş olabilir."
            )
        return text, tokens

    # -- Grid modu (taban sınıfın infer() akışı bunu çağırır) -------------

    def _raw_infer(self, image_path: str, prompt: str) -> tuple[str, int]:
        """Grid JPEG'ini görüntü olarak gönderir (görüntü destekli modelle)."""
        b64 = _encode_jpeg_b64(image_path)
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
        return self._post(self.model_name, content)

    # -- Video modu ------------------------------------------------------

    @staticmethod
    def encode_video(video_path: str) -> str:
        """Videoyu base64'e çevirir ve gövde sınırını denetler.

        Aynı klip üzerinden birden fazla soru sorulacaksa bu metot bir kez
        çağrılıp sonuç tekrar kullanılmalıdır; yeniden kodlama baytları
        değiştirmese de gereksiz iştir, farklı baytlar ise ön ek önbelleğini
        kaçırır.

        Raises:
            FileNotFoundError: Dosya yoksa.
            RuntimeError: Ham dosya gövde sınırını aşarsa.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video bulunamadı: {video_path}")
        raw = open(video_path, "rb").read()
        if len(raw) > _RAW_VIDEO_LIMIT_BYTES:
            raise RuntimeError(
                f"{os.path.basename(video_path)} {len(raw)/1e6:.1f} MB — base64 "
                f"sonrası gövde sınırını aşar. Klip daha kısa segmentlere bölünmeli."
            )
        return base64.b64encode(raw).decode("utf-8")

    def infer_video(
        self,
        video_path: str,
        video_id: str,
        clip_start_sec: float = 0.0,
        clip_end_sec: float = 0.0,
        memory_context: str = "",
        is_segment: bool = False,
        video_b64: str | None = None,
        packet_id: str | None = None,
    ) -> VLMInterpretation:
        """Videoyu doğrudan modele gönderip S8 yorumunu üretir.

        Args:
            video_path: Gönderilecek mp4.
            video_id: İzlenebilirlik kimliği.
            clip_start_sec: Klibin tam videodaki mutlak başlangıcı.
            clip_end_sec: Klibin tam videodaki mutlak bitişi.
            memory_context: Önceki segmentlerin metin hafızası.
            is_segment: Klip daha uzun bir videonun parçası mı.
            video_b64: Önceden kodlanmış base64 içerik (ön ek önbelleği için).
            packet_id: S1b bağlantısı yoksa üretilecek kimlik.

        Returns:
            :class:`contracts.VLMInterpretation` — S8 paketi.
        """
        import uuid

        b64 = video_b64 if video_b64 is not None else self.encode_video(video_path)
        prompt = build_video_system_prompt(
            clip_start_sec, clip_end_sec, memory_context, is_segment
        )
        content = [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
        ]

        t0 = time.perf_counter()
        raw_text, tokens = self._post(self.video_model, content)
        latency_ms = (time.perf_counter() - t0) * 1000

        parse_ok = True
        reasoning = ""
        try:
            reasoning, parsed = _extract_reasoning_and_json(raw_text)
        except ValueError:
            parse_ok = False
            parsed = {
                "scene_summary_tr": "Model çıktısı ayrıştırılamadı.",
                "detected_entities": [],
                "detected_actions_tr": [],
                "risk_events": [],
                "confidence_overall": 0.0,
            }

        entities = [
            DetectedEntity(
                label=e.get("label", ""),
                confidence_hint=e.get("confidence_hint", "low"),
                notes_tr=e.get("notes_tr", ""),
            )
            for e in parsed.get("detected_entities", [])
            if isinstance(e, dict)
        ]

        risk_events = [
            RiskEvent(
                description_tr=r.get("description_tr", ""),
                severity=r.get("severity", "low"),
                confidence=float(r.get("confidence", 0.5)),
                supporting_frame_count=int(r.get("supporting_frame_count", 1)),
                supporting_frame_positions=r.get("supporting_frame_positions", []) or [],
                timestamp_sec=_clamp_timestamp(
                    r.get("timestamp_sec"), clip_start_sec, clip_end_sec
                ),
            )
            for r in parsed.get("risk_events", [])
            if isinstance(r, dict)
        ]

        return VLMInterpretation(
            packet_id=packet_id or str(uuid.uuid4()),
            video_id=video_id,
            model_name=self.video_model,
            model_backend=self.backend_name,
            scene_summary_tr=parsed.get("scene_summary_tr", ""),
            detected_entities=entities,
            detected_actions_tr=parsed.get("detected_actions_tr", []),
            risk_events=risk_events,
            confidence_overall=float(parsed.get("confidence_overall", 0.0) or 0.0),
            computed_confidence=_compute_confidence(parsed, parse_ok),
            inference=InferenceMetrics(latency_ms=latency_ms, tokens_generated=tokens),
            raw_model_output=raw_text,
            reasoning_trace=reasoning,
            segment_start_sec=clip_start_sec,
            segment_end_sec=clip_end_sec,
        )

    def ask_followup(
        self,
        question: str,
        video_b64: str,
        history: list | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Aynı video üzerinden takip sorusu sorar (ön ek önbelleği isabet eder).

        Video baytları değişmediği için bu çağrı soğuk çağrıya göre belirgin
        şekilde hızlıdır. ``history`` verilirse konuşma büyütülerek gönderilir.

        Args:
            question: Sorulacak soru.
            video_b64: :meth:`encode_video` ile bir kez üretilmiş içerik.
            history: Önceki ``{"role", "content"}`` mesajları.
            max_tokens: Bu çağrı için token bütçesi.

        Returns:
            Modelin metin yanıtı.
        """
        messages = list(history) if history else [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Bu videoyu inceleyeceğiz."},
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
            ],
        }]
        messages.append({"role": "user", "content": question})

        payload = {
            "model": self.video_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(), json=payload, timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"].get("content") or ""


def _clamp_timestamp(
    value: object, start_sec: float, end_sec: float
) -> float | None:
    """Modelin verdiği zaman damgasını klip aralığına oturtur.

    Ölçümde model, 10 saniyelik bir klipte ``00:00:55:08`` gibi aralık dışı
    damgalar üretebiliyor. Böyle bir değeri olduğu gibi kabul etmek, olayları
    videonun yanlış anına yerleştirir. Bu yüzden:

    * Sayıya çevrilemeyen değer ``None`` olur (zaman bilinmiyor sayılır).
    * Değer zaten klibin mutlak aralığındaysa olduğu gibi korunur.
    * Mutlak aralığın dışında, ancak klip süresi içindeyse göreli kabul edilip
      mutlak eksene taşınır.
    * Diğer aralık dışı değerler en yakın sınıra çekilir.

    Args:
        value: Modelin ``timestamp_sec`` alanı.
        start_sec: Klibin mutlak başlangıcı.
        end_sec: Klibin mutlak bitişi.

    Returns:
        Aralığa oturtulmuş mutlak saniye veya ``None``.
    """
    if value is None:
        return None
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return None

    duration = max(0.0, end_sec - start_sec)

    # Prompt ve S8 sözleşmesi mutlak tam-video saniyesi ister. Değer zaten
    # segmentin mutlak aralığındaysa göreli sanıp ikinci kez ofsetleme.
    if end_sec > start_sec and start_sec <= ts <= end_sec:
        return ts

    # Mutlak aralığa girmeyen fakat klip süresine uyan değer, modelin prompta
    # rağmen klibe göreli yanıt vermiş olabileceği durumlar için fallback'tir.
    if start_sec > 0 and 0.0 <= ts <= duration + 1e-6:
        ts = start_sec + ts

    if end_sec > start_sec:
        ts = min(max(ts, start_sec), end_sec)
    return ts
