"""Merkezi yapılandırma yönetimi."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class PreprocessingConfig(BaseModel):
    target_frame_count: int = 8
    grid_columns: int = 4
    frame_width: int = 384
    frame_height: int = 216
    channel_a_fps: float = 12.0  # Kanal A yoğun örnekleme hedefi (track sürekliliği)
    use_smart_sampling: bool = True
    ssim_threshold: float = 0.92
    min_laplacian_variance: float = 80.0
    enhance_low_light: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: list[int] = Field(default_factory=lambda: [8, 8])
    critical_frame_count: int = 4  # VLM'e (Kanal B) gidecek kritik kare sayısı


class VehicleLabelingConfig(BaseModel):
    """YOLO 'arac' etiketlerini VLM ile spesifikleştirme (Kanal B öncesi ön adım)."""
    enabled: bool = True
    max_vehicles: int = 8        # tek VLM çağrısında gönderilecek max crop
    min_confidence: float = 0.35  # bu eşiğin altındaki track'ler isimlendirilmez
    padding_ratio: float = 0.15  # bbox genişletme oranı
    max_tokens: int = 768


class PerceptionConfig(BaseModel):
    detector_backend: str = "ultralytics"
    yolo_model: str = "yolov8n.pt"
    custom_classes: list[str] = Field(default_factory=list)
    confidence_threshold: float = 0.35
    tracker: str = "bytetrack"
    tracker_persist: bool = True
    # HF transformers detection backend'i (YOLO eğitilene kadar geçici)
    hf_model: str = "PaddlePaddle/PP-DocLayoutV3_safetensors"
    hf_threshold: float = 0.5
    hf_device: str = "auto"  # "auto" | "cuda" | "cpu" — PP-DocLayoutV3 bazı GPU'larda kararsız, "cpu" güvenli
    vehicle_labeling: VehicleLabelingConfig = Field(default_factory=VehicleLabelingConfig)


class EventThresholds(BaseModel):
    tip_over: dict[str, Any] = Field(default_factory=dict)
    fall: dict[str, Any] = Field(default_factory=dict)
    gathering: dict[str, Any] = Field(default_factory=dict)
    immobility: dict[str, Any] = Field(default_factory=dict)
    ppe_missing: dict[str, Any] = Field(default_factory=dict)
    proximity: dict[str, Any] = Field(default_factory=dict)
    fire_smoke: dict[str, Any] = Field(default_factory=dict)
    leakage: dict[str, Any] = Field(default_factory=dict)


class EventsConfig(BaseModel):
    enabled_rules: list[str] = Field(default_factory=list)
    thresholds: EventThresholds = Field(default_factory=EventThresholds)


class VLLMConfig(BaseModel):
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 8192
    dtype: str = "auto"
    trust_remote_code: bool = True
    max_new_tokens: int = 1024
    temperature: float = 0.15
    top_p: float = 0.9
    repetition_penalty: float = 1.1


class LlamaCppConfig(BaseModel):
    model_repo: str = "mradermacher/TimeLens-7B-i1-GGUF"
    model_file: str = "TimeLens-7B.i1-Q4_K_M.gguf"
    mmproj_repo: str = "mradermacher/TimeLens-7B-GGUF"
    mmproj_file: str = "TimeLens-7B.mmproj-f16.gguf"
    n_ctx: int = 16384
    n_gpu_layers: int = -1
    temperature: float = 0.15
    max_tokens: int = 800
    top_p: float = 0.9
    repeat_penalty: float = 1.15


class TransformersConfig(BaseModel):
    model: str = "llava-hf/LLaVA-NeXT-Video-7B-hf"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "eager"
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.15


class ServerConfig(BaseModel):
    """OpenAI-uyumlu harici çıkarım sunucusu.

    Üç farklı sağlayıcıyı aynı sözleşmeyle kapsar: TEKNOFEST EVREN servisi,
    yerel ``llama-server`` ve ``vllm serve``. Sağlayıcıya özgü kısıtlar
    (görüntü sayısı, zaman aşımı) burada tanımlanır; backend bunları okuyup
    isteği kısıta uyacak şekilde şekillendirir.
    """

    base_url: str = "http://localhost:8080"
    # Görüntü ve metin isteklerinin gideceği model. EVREN'de görüntü kabul eden
    # modeller yalnızca llm-fast ve llm-large'dır ("vlm" görüntüyü reddeder).
    model_name: str = "llm-large"
    # Video (video_url) isteklerinin gideceği model. EVREN'de video analizine
    # özelleşmiş alias "vlm"dir; llm-fast/llm-large da video kabul eder.
    video_model: str = "vlm"
    # API anahtarı doğrudan yapılandırmaya YAZILMAZ; bu ortam değişkeninden
    # okunur (bkz. .env). Böylece anahtar sürüm kontrolüne girmez.
    api_key_env: str = "EVREN_API_KEY"
    # EVREN yığınındaki her katman 1800 sn kullanır. Daha kısa bir istemci
    # zaman aşımı, sunucu isteği işlemeye devam ederken bağlantıyı koparır ve
    # sonuç görüntülenemez (dokümantasyon §hata 06).
    timeout_sec: int = 1800
    # Sağlayıcının istek başına kabul ettiği azami görüntü sayısı. EVREN'de
    # bu değer 2'dir; üçüncü görüntü HTTP 400 döndürür. Backend, kare sayısı
    # bu sınırı aşarsa kareleri tek bir grid görüntüsünde birleştirir.
    max_images_per_request: int = 2
    # Grid birleştirmede kullanılacak sütun sayısı (satır sayısı otomatik).
    grid_columns: int = 4
    # Akıl yürütme (thinking) modu. ``None`` ise parametre hiç gönderilmez
    # (bu uzantıyı desteklemeyen sağlayıcılar için); ``True``/``False`` ise
    # ``chat_template_kwargs.enable_thinking`` olarak iletilir.
    #
    # Varsayılan False'tur, çünkü ölçüm bunu gerektirir: açıkken llm-large'ın
    # skoru 0,900'den 0,580'e DÜŞER ve token maliyeti 17,2 katına çıkar. Daha
    # önemlisi ayrıştırıcı düşünme izini sildiği için, iz token bütçesini
    # tüketirse yanıt HTTP 200 ile birlikte BOŞ döner. Ölçümde kapalıyken
    # üretim 347-373 token arasında kararlıyken, açıkken 627-1321 arasında
    # dalgalanıp bütçeyi taşırabiliyordu.
    enable_thinking: bool | None = False
    temperature: float = 0.15
    # Üst sınırdır, rezervasyon değildir. Bağlam penceresi (262144) prompt ile
    # paylaşıldığından tavana kadar çıkılmaz: 60 sn 720p video ~54k prompt
    # token üretir, 54k + 65k = 119k güvenli marj bırakır.
    max_tokens: int = 65536


class VLMConfig(BaseModel):
    default_backend: str = "auto"
    auto_preference: list[str] = Field(default_factory=lambda: ["vllm", "llama_cpp", "transformers"])
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    llama_cpp: LlamaCppConfig = Field(default_factory=LlamaCppConfig)
    transformers: TransformersConfig = Field(default_factory=TransformersConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


class DecisionAgentConfig(BaseModel):
    system_prompt: str = ""
    include_scene_graph: bool = True
    include_event_signals: bool = True
    include_rag_context: bool = True


class OutputSchemaConfig(BaseModel):
    summary: str = "string"
    events: list[dict[str, Any]] = Field(default_factory=list)
    risk: str = "Düşük | Orta | Yüksek"
    actions: list[str] = Field(default_factory=list)
    reasoning: str = "string"
    confidence: float = 0.0
    triggered_mock_tools: list[dict[str, Any]] = Field(default_factory=list)


class GuardrailConfig(BaseModel):
    max_retries: int = 3
    temperatures: list[float] = Field(default_factory=lambda: [0.15, 0.10, 0.05])
    enable_semantic_check: bool = True
    null_response: str = "Bilmiyorum"


class OutputConfig(BaseModel):
    output_schema: dict[str, Any] = Field(default_factory=dict)
    guardrail: GuardrailConfig = Field(default_factory=GuardrailConfig)


class MetricsConfig(BaseModel):
    enabled: bool = True
    log_inference_time: bool = True
    log_memory_usage: bool = True
    output_json: str = "outputs/metrics.json"


class ProjectConfig(BaseModel):
    name: str = "TEKNOFEST 2026 Senaryo 3"
    version: str = "2.0.0"
    language: str = "tr"
    output_dir: str = "outputs"
    log_dir: str = "logs"


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    decision_agent: DecisionAgentConfig = Field(default_factory=DecisionAgentConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    @field_validator("project", mode="before")
    @classmethod
    def _ensure_project(cls, v: Any) -> Any:
        return v or {}


def _load_dotenv_once() -> None:
    """Depo kökündeki ``.env`` dosyasını ortam değişkenlerine yükler.

    API anahtarları yapılandırma dosyasında değil ``.env``de tutulduğu için
    (bkz. :class:`ServerConfig.api_key_env`), yapılandırma okunurken bu dosyanın
    da yüklenmesi gerekir. Zaten tanımlı olan ortam değişkenleri
    ``override=False`` ile korunur; böylece Docker/CI tarafından verilen
    değerler dosya tarafından ezilmez.

    ``python-dotenv`` kurulu değilse sessizce atlanır: bu durumda anahtarın
    ortamda elle tanımlanmış olması gerekir.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def load_config(path: str | Path | None = None) -> AppConfig:
    """config.yaml'yi yükler; yoksa varsayılanları döner.

    Yapılandırmayla birlikte depo kökündeki ``.env`` dosyası da ortama
    yüklenir; API anahtarları oradan okunur.
    """
    _load_dotenv_once()

    if path is None:
        path = os.environ.get("TEKNOFEST_CONFIG", "config.yaml")
    path = Path(path)

    raw: dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    return AppConfig(**raw)


def get_data_path(filename: str) -> Path:
    return Path("data") / filename
