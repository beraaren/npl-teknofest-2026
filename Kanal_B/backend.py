"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DOSYA: backend.py                                                           ║
║  KATMAN: VLM Inference — plan/05 Karar Ajanı ve VLM Backend                 ║
║  ROL   : S1b paketini (grid görseli) alır, VLM modeline gönderir,           ║
║          Türkçe sahne yorumunu S8 sözleşmesine dönüştürür.                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

VERİ AKIŞI İÇİNDEKİ YERİ:
  preprocessing.py → [VLMFramePacket S1b]
    → VLMBackendBase.infer()
      → _raw_infer() [modele gönder]
      → _extract_json() [yanıtı ayrıştır]
      → VLMInterpretation [S8 paketi oluştur]
    → Karar Ajanı'na iletilir

BACKEND ADAPTER PATTERN:
  VLMBackendBase (ABC) ← ortak arayüz
    ├── VLLMBackend      : vLLM sunucusu (OpenAI-uyumlu API, GPU hızlı üretim için)
    ├── LlamaCppBackend  : llama.cpp sunucusu (GGUF model, CPU veya küçük GPU için)
    └── TransformersBackend: HuggingFace transformers (sunucusuz, tek prosesli test)

  Karar Ajanı hangi backend'in çalıştığını BILMEZ → sadece VLMInterpretation alır.
  Backend seçimi config.yaml'daki [vlm.default_backend] ile yapılır.

SISTEM PROMPT STRATEJİSİ:
  SYSTEM_PROMPT → "system" rolü olarak gönderilir (S6 değişikliği).
  User mesajı SADECE grid görselini içerir.
  Bu sayede model talimatları ve görsel girdi ayrı tutulur.
  ⚠️ Bazı modeller (eski LLaVA GGUF) system rolünü desteklemeyebilir.

HATA TOLERANSI:
  1. JSON ayrıştırma başarısız → S8 sözleşmesi bozulmadan boş/düşük güvenli fallback
  2. HTTP hatası → requests.raise_for_status() ile yukarı taşınır
  3. Model boş yanıt → fallback parsed dict ile devam
"""
from __future__ import annotations
import base64
import json
import re
import time
from abc import ABC, abstractmethod

import requests

# S1b ve S8 sözleşme yapıları
from contracts import (
    VLMInterpretation, DetectedEntity, InferenceMetrics, VLMFramePacket,
)


# ---------------------------------------------------------------------------
# Sistem Promptu — model davranışını yönlendiren temel talimat
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Sen bir güvenlik kamerası sahne analistisin. Sana bir video akışından alınmış "
    "kareleri içeren tek bir ızgara (grid) görseli veriliyor. Görevin, olay motorundan "
    "ve nesne dedektöründen BAĞIMSIZ olarak, gördüğün sahneyi kendi gözünle yorumlaman. "
    "SADECE aşağıdaki JSON şemasına uyan, başka hiçbir metin içermeyen bir çıktı üret:\n"
    "{\n"
    '  "scene_summary_tr": "1-3 cümlelik Türkçe sahne özeti",\n'
    '  "detected_entities": [{"label": "kısa etiket", "confidence_hint": "low|medium|high", "notes_tr": "kısa not"}],\n'
    '  "detected_actions_tr": ["gözlemlenen eylemler, Türkçe"],\n'
    '  "risk_flags_tr": ["varsa riskli/anormal unsurlar, Türkçe; yoksa boş liste"],\n'
    '  "confidence_overall": 0.0\n'
    "}"
    # NEDEN JSON ŞEMASINİ PROMPT'A EKLİYORUZ?
    # Model çıktısını S8 sözleşmesine uydurabilmek için.
    # _extract_json() bu JSON'ı ayrıştırıp VLMInterpretation'a dönüştürür.
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

    S8 DEĞİŞİKLİK: Eski tek kademeli kırılgan regex yerine 3 kademeli strateji.
    Model farklı formatlarda yanıt üretebilir; her durumu yakalar:

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

    # 1. Kademe: direkt parse (modelin ideal davranışı)
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


# ---------------------------------------------------------------------------
# Soyut temel sınıf — ortak inference arayüzü
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
          1. _raw_infer() çağrısını zamanla (latency_ms hesabı için)
          2. JSON çıktısını 3 kademeli parse ile ayrıştır
          3. Parse başarısız → S8 sözleşmesini bozmadan fallback dict kullan
          4. VLMInterpretation (S8) oluştur ve döndür
        """
        t0 = time.perf_counter()
        raw_text, tokens = self._raw_infer(packet.grid_image_path, SYSTEM_PROMPT)
        latency_ms = (time.perf_counter() - t0) * 1000  # saniye → milisaniye

        try:
            parsed = _extract_json(raw_text)
        except ValueError:
            # Model şemaya uymayan çıktı verdiyse S8 sözleşmesini bozmadan devam et.
            # Karar Ajanı yine de beklediği alanları bulsun (confidence=0.0 ile uyarı verir).
            parsed = {
                "scene_summary_tr": "Model çıktısı ayrıştırılamadı.",
                "detected_entities": [],
                "detected_actions_tr": [],
                "risk_flags_tr": [],
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
        ]

        # S8 sözleşme paketi — Karar Ajanı'nın 3. girdisi
        return VLMInterpretation(
            packet_id=packet.packet_id,       # S1b ile izlenebilirlik bağlantısı
            video_id=packet.video_id,
            model_name=self.model_name,
            model_backend=self.backend_name,
            scene_summary_tr=parsed.get("scene_summary_tr", ""),
            detected_entities=entities,
            detected_actions_tr=parsed.get("detected_actions_tr", []),
            risk_flags_tr=parsed.get("risk_flags_tr", []),
            confidence_overall=float(parsed.get("confidence_overall", 0.0)),
            inference=InferenceMetrics(latency_ms=latency_ms, tokens_generated=tokens),
            raw_model_output=raw_text,        # debug/audit için ham çıktı
        )


# ---------------------------------------------------------------------------
# Backend 1 — vLLM (GPU hızlı üretim, OpenAI-uyumlu API)
# ---------------------------------------------------------------------------

class VLLMBackend(VLMBackendBase):
    """vLLM OpenAI-uyumlu sunucu ile inference.

    KULLANIM KOŞULU:
      vLLM sunucusu görsel destekli bir model ile ayağa kaldırılmış olmalı.
      Örnek: vllm serve Qwen/Qwen2.5-VL-7B-Instruct --tensor-parallel-size 1

    MESAJ YAPISI (S6 değişikliği):
      system: SYSTEM_PROMPT (talimatlar)
      user  : grid görseli (base64 data URI)

      ⚠️ Model system rolünü desteklemiyorsa system bloğunu kaldır,
         prompt'u user content'inin başına {"type":"text","text":prompt} olarak ekle.
    """
    backend_name = "vllm"

    def __init__(self, base_url: str, model_name: str,
                 max_tokens: int = 512, temperature: float = 0.2):
        # S5: temperature artık __init__ parametresi → config.yaml'dan gelir
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature  # config.yaml'da vllm.temperature: 0.15

    def _raw_infer(self, image_path: str, prompt: str) -> tuple[str, int]:
        b64 = _encode_jpeg_b64(image_path)
        payload = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,  # S5: hardcode 0.2 → self.temperature
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
        resp = requests.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=60)
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

    ⚠️ LLaVA tabanlı bazı GGUF modeller system rolünü desteklemez.
       Desteklemiyorsa bu sınıftaki messages'tan system bloğunu kaldır ve
       prompt'u user content'e text olarak ekle:
       {"type": "text", "text": prompt}
    """
    backend_name = "llama.cpp"

    def __init__(self, base_url: str, model_name: str,
                 max_tokens: int = 512, temperature: float = 0.2):
        # S5: temperature artık __init__ parametresi → config.yaml'dan gelir
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature  # config.yaml'da llama_cpp.temperature: 0.15

    def _raw_infer(self, image_path: str, prompt: str) -> tuple[str, int]:
        b64 = _encode_jpeg_b64(image_path)
        payload = {
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
        resp = requests.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=60)
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

    if backend == "vllm":
        # S5: temperature config'den geçiyor (config.yaml: vllm.temperature: 0.15)
        return VLLMBackend(
            config["base_url"], config["model_name"],
            config.get("max_tokens", 512), config.get("temperature", 0.15),
        )

    if backend == "llama.cpp":
        return LlamaCppBackend(
            config["base_url"], config["model_name"],
            config.get("max_tokens", 512), config.get("temperature", 0.15),
        )

    if backend == "transformers":
        return TransformersBackend(
            config["model_name"],
            config.get("device", "cuda"),
            config.get("max_tokens", 512),
        )

    raise ValueError(f"Bilinmeyen backend: {backend!r} — 'vllm', 'llama.cpp' veya 'transformers' olmalı")