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
    parser.add_argument("--backend", type=str, default=None, help="VLM backend: vllm|llama_cpp|transformers|server")
    parser.add_argument("--detector", type=str, default=None, help="Tespit backend'i: ultralytics|hf_transformers")
    parser.add_argument("--output", type=str, default=None, help="Çıktı JSON yolu")
    parser.add_argument("--no-enhance", action="store_true", help="Görsel iyileştirmeyi devre dışı bırak")
    parser.add_argument("--save-grid", action="store_true", help="VLM'e gönderilen grid'i kaydet")
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Deney adı: outputs/experiments/<isim>/ altına tarih+config-hash ile kaydeder (A/B test arşivi için)",
    )
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


def _short_text(value: Any, limit: int = 400) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
    if len(text) > limit:
        return text[:limit] + f"\n... ({len(text)} karakter, kısaltıldı)"
    return text


def main(args=None) -> None:
    # Lazy imports: böylece --help veya temel kontroller bağımlılık kurulmadan çalışır.
    import numpy as np
    from PIL import Image

    from .config import AppConfig, load_config
    from .events.event_engine import EventEngine
    from .models.vlm_backend import create_backend
    from .output.guardrail import OutputGuardrail
    from .perception.observer_agent import ObserverAgent
    from .perception.vehicle_labeler import apply_vehicle_labels, label_vehicles
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
    if args.detector:
        config.perception.detector_backend = args.detector

    logger = get_logger("main", config.project.log_dir)
    logger.info(f"{config.project.name} v{config.project.version} başlatıldı.")
    logger.info(f"Girdi video: {args.video}")
    # NOT: Burada 'config.vlm.backend' okunuyordu; VLMConfig'de böyle bir alan
    # yok, doğrusu 'default_backend'. Hata yalnızca --backend VERİLMEDİĞİNDE
    # ortaya çıkıyordu: 'or' kısa devre yaptığı için --backend verildiğinde sağ
    # taraf hiç değerlendirilmiyor ve AttributeError gizleniyordu.
    logger.info(f"Seçilen backend: {args.backend or config.vlm.default_backend}")
    logger.info(f"Seçilen detector: {args.detector or config.perception.detector_backend}")

    metrics = MetricsCollector(config.metrics.output_json)

    # Çıktı dizinleri
    out_dir = Path(config.project.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Video oku — Kanal A: yoğun örnekleme (track sürekliliği için)
    # ------------------------------------------------------------------
    with metrics.measure("video_read"):
        reader = VideoReader(args.video)
        native_fps = reader.fps
        target_fps = config.preprocessing.channel_a_fps
        step = max(1, round(native_fps / target_fps)) if target_fps > 0 else 1
        fps_a = native_fps / step  # Kanal A'nın efektif fps'i (zaman kuralları buna göre)
        logger.info(
            f"Video: {args.video} | {reader.total_frames} kare | {native_fps:.2f} fps"
            f" → Kanal A: her {step}. kare ({fps_a:.1f} fps)"
        )
        channel_a_frames, channel_a_indices = [], []
        for i, frame in enumerate(reader.iter_frames()):
            if i % step == 0:
                channel_a_frames.append(frame)
                channel_a_indices.append(i)
        reader.close()
        logger.info(
            f"Video özeti: toplam_kare={reader.total_frames}, fps={native_fps:.2f}, "
            f"Kanal A kare_sayısı={len(channel_a_frames)}, örnekleme_adımı={step}, Kanal_A_fps={fps_a:.1f}"
        )

    # ------------------------------------------------------------------
    # 2. Kanal B: akıllı örnekleme (Kanal A akışından 8 kare)
    # ------------------------------------------------------------------
    with metrics.measure("preprocessing"):
        sampler = FrameSampler(
            target_count=config.preprocessing.target_frame_count,
            use_smart_sampling=config.preprocessing.use_smart_sampling,
            ssim_threshold=config.preprocessing.ssim_threshold,
            min_laplacian_variance=config.preprocessing.min_laplacian_variance,
        )
        sampled_frames, sampled_pos = sampler.sample(channel_a_frames, len(channel_a_frames))
        sampled_indices = [channel_a_indices[p] for p in sampled_pos]  # gerçek video indeksleri

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
        logger.info(
            f"Ön işleme çıktısı: seçilen_kare_sayısı={len(sampled_frames)}, "
            f"indeksler={sampled_indices}, boyut={target_size[0]}x{target_size[1]}"
        )

    # ------------------------------------------------------------------
    # 3. Gözlemci Ajan — Kanal A'nın tamamı üzerinde (yoğun akış)
    # ------------------------------------------------------------------
    with metrics.measure("perception"):
        observer = ObserverAgent(config.perception)
        observations = observer.observe_video(channel_a_frames, native_fps, sampled_indices=channel_a_indices)
        scene_graphs = [obs["scene_graph"] for obs in observations]
        total_nodes = sum(len(obs.get("scene_graph", {}).get("nodes", [])) for obs in observations)
        logger.info(
            f"Algılama çıktısı: gözlem_sayısı={len(observations)}, toplam_nesne_düğümü={total_nodes}"
        )

    # ------------------------------------------------------------------
    # 3b. Araç isimlendirme — YOLO 'arac' etiketlerini VLM ile spesifikleştir.
    # Kanal B / karar çağrılarından ÖNCE çalışır; kural motoru spesifik isimleri
    # CanonicalClass.normalize() ile yine 'arac'a indirgediği için kurallar bozulmaz.
    # Backend burada bir kez oluşturulur, adım 7'de karar ajanına aynısı geçilir.
    # ------------------------------------------------------------------
    backend = None
    if config.perception.vehicle_labeling.enabled:
        with metrics.measure("vehicle_labeling"):
            backend = create_backend(config.vlm, force=args.backend)
            logger.info(f"Kullanılan VLM backend: {backend.name()}")
            label_map = label_vehicles(
                observer.tracks, channel_a_frames, backend, config.perception.vehicle_labeling
            )
            labeled_count = apply_vehicle_labels(observer.tracks, observations, label_map)
            logger.info(f"Araç isimlendirme çıktısı: {labeled_count} araç spesifik etiket aldı")

    # ------------------------------------------------------------------
    # 4. Olay Tespit Motoru — zaman kuralları Kanal A fps'iyle
    # ------------------------------------------------------------------
    with metrics.measure("event_engine"):
        event_engine = EventEngine(config.events, fps=fps_a)
        for obs in observations:
            event_engine.process_observation(obs)
        event_signals = event_engine.get_signals()
        signal_preview = [
            {"event_type": s.get("event_type"), "timestamp": s.get("timestamp")}
            for s in event_signals[:5]
        ]
        logger.info(
            f"Olay sinyali çıktısı: sinyal_sayısı={len(event_signals)}, ilk_sinyaller={signal_preview}"
        )

    # ------------------------------------------------------------------
    # 4b. Kanal B için kritik kare seçimi
    # ------------------------------------------------------------------
    with metrics.measure("critical_frames"):
        critical_frames, critical_indices = select_critical_frames(
            sampled_frames,
            sampled_indices,
            event_signals,
            fps=native_fps,
            max_count=config.preprocessing.critical_frame_count,
        )
        logger.info(
            f"Kritik kare çıktısı: kritik_kare_sayısı={len(critical_frames)}, indeksler={critical_indices}"
        )

    # ------------------------------------------------------------------
    # 5. RAG Katmanı
    # ------------------------------------------------------------------
    with metrics.measure("rag"):
        rag = RAGLayer()
        # Ana sorgu: Observer raporu; event_signals ikincil filtre/boost
        rag_context = rag.build_context(observations, event_signals)
        logger.info(
            "RAG çıktısı: "
            f"risk_level={rag_context.get('risk_level')}, "
            f"risk_score={rag_context.get('risk_score')}, "
            f"matched_patterns={rag_context.get('matched_patterns', [])}, "
            f"actions={rag_context.get('actions', [])}"
        )

    # ------------------------------------------------------------------
    # 6. Hafıza ve Mock Tool'lar
    # ------------------------------------------------------------------
    memory = ShortTermMemory()
    for sig in event_signals:
        memory.add(sig, entry_type="event", timestamp=_time_to_seconds(sig.get("timestamp", "00:00")))

    tools = MockToolRegistry()
    logger.info(f"Hafıza çıktısı: {_short_text(memory.to_prompt_context(), limit=600)}")
    logger.info(f"Mock araçlar: {', '.join(sorted(tools.tools.keys()))}")

    # ------------------------------------------------------------------
    # 7. Karar Ajanı (VLM)
    # ------------------------------------------------------------------
    with metrics.measure("vlm_decision"):
        if backend is None:  # araç isimlendirme kapalıysa burada oluştur
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

        # Kanal B: bağımsız VLM kanalı (S8). Kanal_B paketi videoyu KENDİ
        # ön işlemesiyle işler — ona olay sinyali, RAG, kritik kare gibi
        # başka kanal çıktısı VERİLMEZ. Birleştirme yalnızca karar ajanında.
        vlm_interpretation = None
        try:
            import sys

            # Kanal_B modülleri birbirini paket öneki olmadan içe aktarır
            # (`from vlm_backend import ...`), bu yüzden Kanal_B dizininin
            # KENDİSİ sys.path'te olmalıdır — yalnızca proje kökü yetmez.
            # test_akis.py de aynı deseni kullanır.
            project_root = Path(__file__).resolve().parent.parent
            for entry in (str(project_root), str(project_root / "Kanal_B")):
                if entry not in sys.path:
                    sys.path.insert(0, entry)
            from pipeline import run_channel_b  # Kanal_B/pipeline.py

            vlm_interpretation = run_channel_b(
                args.video,
                video_id=Path(args.video).stem,
                output_dir=str(out_dir / "channel_b"),
            )
        except Exception as e:
            logger.warning(f"Kanal_B paketi çalışmadı ({e}); interpret_frames'e düşülüyor.")
            vlm_interpretation = agent.interpret_frames(sampled_frames)

        logger.info(f"Kanal B çıktısı: {_short_text(vlm_interpretation, limit=1200)}")

        decision_raw = agent.decide(
            event_signals=event_signals,
            scene_graphs=scene_graphs,
            rag_context=rag_context,
            vlm_interpretation=vlm_interpretation,
        )
        logger.info(f"Karar ajanı ham çıktısı: {_short_text(decision_raw.get('raw_text', ''), limit=1600)}")

    # ------------------------------------------------------------------
    # 8. Guardrail
    # ------------------------------------------------------------------
    with metrics.measure("guardrail"):
        guardrail = OutputGuardrail(config.output.guardrail)

        final_output = guardrail.validate(
            decision_raw["raw_text"],
            decision_raw["retry_fn"],
            rag_risk_level=decision_raw["rag_risk_level"],
        )
        logger.info(
            "Guardrail çıktısı: "
            f"risk={final_output.get('risk')}, "
            f"confidence={final_output.get('confidence')}, "
            # 'confidence_word' alanı AnalysisOutput şemasında yok; her zaman
            # None basıyordu, bu yüzden log satırından çıkarıldı.
            f"summary={final_output.get('summary')}, "
            f"actions={final_output.get('actions', [])}"
        )

    # ------------------------------------------------------------------
    # 9. Mock tool'ları zenginleştir ve ÇALIŞTIR
    # ------------------------------------------------------------------
    triggered = final_output.get("triggered_mock_tools", [])
    if not triggered:
        suggested = tools.suggest_tools(final_output["risk"], [e.get("event_type", "") for e in final_output.get("events", [])])
        final_output["triggered_mock_tools"] = [
            {"tool_name": s["tool_name"], "params": {"location": "saha", "reason": final_output["summary"][:100]}}
            for s in suggested
        ]
        logger.info(f"Önerilen araçlar: {final_output['triggered_mock_tools']}")

    # Seçilen araçları gerçekten çalıştır ve sonuçları kaydet
    tool_results = []
    logger.info(f"Çalıştırılacak araçlar: {final_output['triggered_mock_tools']}")
    for tool_call in final_output["triggered_mock_tools"]:
        result = tools.execute(tool_call["tool_name"], tool_call["params"])
        logger.info(f"Tool sonucu: {tool_call['tool_name']} → {result}")
        tool_results.append(result)
    final_output["tool_execution_results"] = tool_results

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
        "vlm_interpretation": vlm_interpretation,
        "geometric_signals": event_signals,
        # Olay zaman damgalarını öne çıkar (deneme.py'deki başlangıç/bitiş mantığıyla uyumlu)
        "event_timestamps": [
            {
                "event_type": sig.get("event_type", ""),
                "timestamp": sig.get("timestamp", ""),
                "seconds": _time_to_seconds(sig.get("timestamp", "00:00")),
            }
            for sig in event_signals
        ],
    }

    if args.experiment:
        import datetime
        import hashlib
        exp_dir = out_dir / "experiments" / args.experiment
        exp_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg_hash = hashlib.md5(open(args.config, "rb").read()).hexdigest()[:6]
        output_path = exp_dir / f"result_{ts}_{cfg_hash}.json"
    else:
        output_path = Path(args.output) if args.output else out_dir / "analysis_result.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    logger.info(f"Sonuç JSON çıktısı: {_short_text(final_output, limit=2000)}")
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
