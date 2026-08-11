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
    """OpenAI-uyumlu harici sunucu (llama-server / vllm serve) — Kanal_B ile aynı sözleşme."""
    base_url: str = "http://localhost:8080"
    model_name: str = "llava-v1.6-mistral-7b"
    temperature: float = 0.15
    max_tokens: int = 800


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


def load_config(path: str | Path | None = None) -> AppConfig:
    """config.yaml'yi yükler; yoksa varsayılanları döner."""
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
