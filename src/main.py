"""TEKNOFEST 2026 Senaryo 3 — Ana pipeline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TEKNOFEST 2026 Senaryo 3 — Video Analiz Ajanı")
    parser.add_argument("--video", type=str, default="video.mp4", help="Analiz edilecek video dosyası")
    parser.add_argument("--config", type=str, default="config.yaml", help="config.yaml yolu")
    parser.add_argument("--backend", type=str, default=None, help="VLM backend: vllm|llama_cpp|transformers")
    parser.add_argument("--output", type=str, default=None, help="Çıktı JSON yolu")
    parser.add_argument("--no-enhance", action="store_true", help="Görsel iyileştirmeyi devre dışı bırak")
    parser.add_argument("--save-grid", action="store_true", help="VLM'e gönderilen grid'i kaydet")
    return parser


def save_grid_image(frames: List[Any], cols: int, path: Path) -> None:
    import numpy as np
    from PIL import Image

    rows = (len(frames) + cols - 1) // cols
    h, w = frames[0].shape[:2]
    grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    for idx, f in enumerate(frames):
        r, c = divmod(idx, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
    Image.fromarray(grid).save(path)


def main(args=None) -> None:
    # Lazy imports: böylece --help veya temel kontroller bağımlılık kurulmadan çalışır.
    import numpy as np
    from PIL import Image

    from .config import AppConfig, load_config
    from .events.event_engine import EventEngine
    from .models.vlm_backend import create_backend
    from .output.guardrail import OutputGuardrail
    from .perception.observer_agent import ObserverAgent
    from .preprocessing.critical_frames import select_critical_frames
    from .preprocessing.enhancer import LowLightEnhancer
    from .preprocessing.frame_sampler import FrameSampler
    from .preprocessing.video_reader import VideoReader
    from .reasoning.decision_agent import DecisionAgent
    from .reasoning.memory import ShortTermMemory
    from .reasoning.mock_tools import MockToolRegistry
    from .reasoning.rag_layer import RAGLayer
    from .utils.logger import get_logger
    from .utils.timing import MetricsCollector

    if args is None:
        args = build_parser().parse_args()
    config: AppConfig = load_config(args.config)

    logger = get_logger("main", config.project.log_dir)
    logger.info(f"{config.project.name} v{config.project.version} başlatıldı.")

    metrics = MetricsCollector(config.metrics.output_json)

    # Çıktı dizinleri
    out_dir = Path(config.project.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Video oku ve kareleri çıkar
    # ------------------------------------------------------------------
    with metrics.measure("video_read"):
        reader = VideoReader(args.video)
        logger.info(f"Video: {args.video} | {reader.total_frames} kare | {reader.fps:.2f} fps")
        frames = list(reader.iter_frames())
        reader.close()

    # ------------------------------------------------------------------
    # 2. Akıllı örnekleme
    # ------------------------------------------------------------------
    with metrics.measure("preprocessing"):
        sampler = FrameSampler(
            target_count=config.preprocessing.target_frame_count,
            use_smart_sampling=config.preprocessing.use_smart_sampling,
            ssim_threshold=config.preprocessing.ssim_threshold,
            min_laplacian_variance=config.preprocessing.min_laplacian_variance,
        )
        sampled_frames, sampled_indices = sampler.sample(frames, reader.total_frames)
        logger.info(f"Seçilen kare indeksleri: {sampled_indices}")

        # Görsel iyileştirme
        if not args.no_enhance and config.preprocessing.enhance_low_light:
            enhancer = LowLightEnhancer(
                enabled=True,
                clip_limit=config.preprocessing.clahe_clip_limit,
                grid_size=tuple(config.preprocessing.clahe_grid_size),
            )
            sampled_frames = [enhancer.enhance(f) for f in sampled_frames]

        # Resize
        target_size = (config.preprocessing.frame_width, config.preprocessing.frame_height)
        sampled_frames = [np.array(Image.fromarray(f).resize(target_size)) for f in sampled_frames]

    # ------------------------------------------------------------------
    # 3. Gözlemci Ajan
    # ------------------------------------------------------------------
    with metrics.measure("perception"):
        observer = ObserverAgent(config.perception)
        observations = observer.observe_video(sampled_frames, reader.fps, sampled_indices=sampled_indices)
        scene_graphs = [obs["scene_graph"] for obs in observations]

    # ------------------------------------------------------------------
    # 4. Olay Tespit Motoru
    # ------------------------------------------------------------------
    with metrics.measure("event_engine"):
        event_engine = EventEngine(config.events, fps=reader.fps)
        for obs in observations:
            event_engine.process_observation(obs)
        event_signals = event_engine.get_signals()
        logger.info(f"Tespit edilen geometrik olay sinyalleri: {len(event_signals)}")

    # ------------------------------------------------------------------
    # 4b. Kanal B için kritik kare seçimi
    # ------------------------------------------------------------------
    with metrics.measure("critical_frames"):
        critical_frames, critical_indices = select_critical_frames(
            sampled_frames,
            sampled_indices,
            event_signals,
            fps=reader.fps,
            max_count=config.preprocessing.critical_frame_count,
        )
        logger.info(f"Kritik kare indeksleri: {critical_indices}")

    # ------------------------------------------------------------------
    # 5. RAG Katmanı
    # ------------------------------------------------------------------
    with metrics.measure("rag"):
        rag = RAGLayer()
        rag_context = rag.build_context(event_signals)

    # ------------------------------------------------------------------
    # 6. Hafıza ve Mock Tool'lar
    # ------------------------------------------------------------------
    memory = ShortTermMemory()
    for sig in event_signals:
        memory.add(sig, entry_type="event", timestamp=_time_to_seconds(sig.get("timestamp", "00:00")))

    tools = MockToolRegistry()

    # ------------------------------------------------------------------
    # 7. Karar Ajanı (VLM)
    # ------------------------------------------------------------------
    with metrics.measure("vlm_decision"):
        backend = create_backend(config.vlm, force=args.backend)
        logger.info(f"Kullanılan VLM backend: {backend.name()}")

        agent = DecisionAgent(
            config=config.decision_agent,
            vlm_config=config.vlm,
            rag=rag,
            memory=memory,
            tools=tools,
            backend=backend,
        )

        # Kanal B: kritik karelerin bağımsız, genel betimlemesi
        vlm_observation = agent.interpret_frames(critical_frames)

        decision_result = agent.decide(
            images=sampled_frames,
            event_signals=event_signals,
            scene_graphs=scene_graphs,
            fps=reader.fps,
            vlm_observation=vlm_observation,
        )

    # ------------------------------------------------------------------
    # 8. Guardrail
    # ------------------------------------------------------------------
    with metrics.measure("guardrail"):
        guardrail = OutputGuardrail(config.output.guardrail)

        def retry_generate(temp: float) -> str:
            return backend.generate(
                sampled_frames,
                agent._build_prompt(
                    event_signals,
                    scene_graphs,
                    rag_context,
                    memory.to_prompt_context(),
                    vlm_observation,
                ),
                temperature=temp,
                max_tokens=config.vlm.vllm.max_new_tokens,
            )

        final_output = guardrail.validate(
            decision_result["raw"],
            retry_generate,
            rag_risk_level=rag_context.get("risk_level", "Düşük"),
        )

    # ------------------------------------------------------------------
    # 9. Mock tool'ları zenginleştir
    # ------------------------------------------------------------------
    triggered = final_output.get("triggered_mock_tools", [])
    if not triggered:
        suggested = tools.suggest_tools(final_output["risk"], [e.get("event_type", "") for e in final_output.get("events", [])])
        final_output["triggered_mock_tools"] = [
            {"tool_name": s["tool_name"], "params": {"location": "saha", "reason": final_output["summary"][:100]}}
            for s in suggested
        ]

    # ------------------------------------------------------------------
    # 10. Kaydet
    # ------------------------------------------------------------------
    final_output["metadata"] = {
        "video": args.video,
        "total_frames": reader.total_frames,
        "fps": reader.fps,
        "sampled_indices": sampled_indices,
        "critical_indices": critical_indices,
        "vlm_backend": backend.name(),
        "vlm_observation": vlm_observation,
        "geometric_signals": event_signals,
    }

    output_path = Path(args.output) if args.output else out_dir / "analysis_result.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    logger.info(f"Sonuç kaydedildi: {output_path}")

    if args.save_grid:
        grid_path = out_dir / "vlm_input_grid.jpg"
        save_grid_image(sampled_frames, config.preprocessing.grid_columns, grid_path)
        logger.info(f"VLM grid kaydedildi: {grid_path}")

    metrics.add("event_count", len(event_signals))
    metrics.add("risk_level", final_output["risk"])
    metrics.save()

    logger.info("Analiz tamamlandı.")


def _time_to_seconds(ts: str) -> float:
    try:
        parts = ts.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except Exception:
        pass
    return 0.0


if __name__ == "__main__":
    _args = build_parser().parse_args()
    main(_args)
