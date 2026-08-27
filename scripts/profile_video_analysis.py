#!/usr/bin/env python3
"""Tek bir video için detaylı performans ve darboğaz (profiling) analizi yapar."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "Kanal_B")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env", override=True)

import numpy as np
from PIL import Image
import torch

from src.config import load_config
from src.preprocessing.video_reader import VideoReader
from src.preprocessing.frame_sampler import FrameSampler
from src.preprocessing.enhancer import LowLightEnhancer
from src.perception.observer_agent import ObserverAgent
from src.events.event_engine import EventEngine
from src.reasoning.rag_layer import RAGLayer
from src.reasoning.memory import ShortTermMemory
from src.reasoning.mock_tools import MockToolRegistry
from src.reasoning.decision_agent import DecisionAgent
from src.models.vlm_backend import create_backend
from src.output.guardrail import OutputGuardrail
from Kanal_B.preprocessing import probe_video
from Kanal_B.pipeline import run_channel_b


def profile_video(video_path: Path, config_path: Path = REPO_ROOT / "config.yaml"):
    print(f"\n{'='*70}")
    print(f"📊 PERFORMANS VE DARBOĞAZ PROFİLLEME BAŞLADI: {video_path.name}")
    print(f"{'='*70}")
    
    timings = {}
    total_start = time.perf_counter()

    # 0. Config
    t0 = time.perf_counter()
    config = load_config(str(config_path))
    timings["0. Config Yükleme"] = time.perf_counter() - t0

    # 1. Video Probe
    t0 = time.perf_counter()
    info = probe_video(str(video_path))
    duration_sec = float(info["duration_sec"])
    timings["1. Video Probe"] = time.perf_counter() - t0
    print(f"📁 Video Süresi: {duration_sec:.1f}s | Çözünürlük: {info['width']}x{info['height']} | FPS: {info['fps']}")

    # 2. Frame Decoding (VideoReader)
    t0 = time.perf_counter()
    reader = VideoReader(str(video_path))
    native_fps = reader.fps or 25.0
    target_fps = config.preprocessing.channel_a_fps
    step = max(1, round(native_fps / target_fps)) if target_fps > 0 else 1
    fps_a = native_fps / step

    channel_a_frames, channel_a_indices = [], []
    for idx, frame in enumerate(reader.iter_frames()):
        if idx % step == 0:
            channel_a_frames.append(frame)
            channel_a_indices.append(idx)
    reader.close()
    timings["2. Video Okuma & Kare Çıkarma (Kanal A)"] = time.perf_counter() - t0
    print(f"🖼️ Okunan Kare Sayısı: {len(channel_a_frames)} (Örnekleme FPS: {fps_a:.1f})")

    # 3. Frame Sampler & Preprocessing (CLAHE / SSIM / Laplacian)
    t0 = time.perf_counter()
    sampler = FrameSampler(
        target_count=config.preprocessing.target_frame_count,
        use_smart_sampling=config.preprocessing.use_smart_sampling,
        ssim_threshold=config.preprocessing.ssim_threshold,
        min_laplacian_variance=config.preprocessing.min_laplacian_variance,
    )
    sampled_frames, sampled_pos = sampler.sample(channel_a_frames, len(channel_a_frames))
    sampled_indices = [channel_a_indices[p] for p in sampled_pos]
    if config.preprocessing.enhance_low_light:
        enhancer = LowLightEnhancer(
            enabled=True,
            clip_limit=config.preprocessing.clahe_clip_limit,
            grid_size=tuple(config.preprocessing.clahe_grid_size),
        )
        sampled_frames = [enhancer.enhance(f) for f in sampled_frames]
    target_size = (config.preprocessing.frame_width, config.preprocessing.frame_height)
    sampled_frames = [np.array(Image.fromarray(f).resize(target_size)) for f in sampled_frames]
    timings["3. Frame Sampler & CLAHE (Yedek Kareler)"] = time.perf_counter() - t0

    # 4. ObserverAgent (YOLO + Tracker)
    t0 = time.perf_counter()
    observer = ObserverAgent(config.perception)
    t_model_load = time.perf_counter() - t0
    
    t0_infer = time.perf_counter()
    observations = observer.observe_video(
        channel_a_frames, native_fps, sampled_indices=channel_a_indices
    )
    t_infer = time.perf_counter() - t0_infer
    scene_graphs = [obs["scene_graph"] for obs in observations]
    timings["4.1 YOLO Model Yükleme"] = t_model_load
    timings["4.2 YOLO Çıkarım + ByteTrack Takip"] = t_infer

    # 5. Event Engine (Kural Motoru)
    t0 = time.perf_counter()
    engine = EventEngine(config.events, fps=fps_a)
    for obs in observations:
        engine.process_observation(obs)
    event_signals = engine.get_signals()
    timings["5. İSG Kural Motoru (Event Engine)"] = time.perf_counter() - t0

    # 6. RAG & Hafıza
    t0 = time.perf_counter()
    rag = RAGLayer()
    rag_context = rag.build_context(observations, event_signals)
    memory = ShortTermMemory()
    for sig in event_signals:
        memory.add(sig, entry_type="event")
    tools = MockToolRegistry()
    timings["6. RAG Katmanı & Bellek Kurulumu"] = time.perf_counter() - t0

    # 7. Kanal B: EVREN VLM Video Yorumu
    channel_b_dir = REPO_ROOT / "data" / "library" / "channel_b" / f"profile_{video_path.stem}"
    t0 = time.perf_counter()
    vlm_interpretation = run_channel_b(
        str(video_path), video_id=f"profile_{video_path.stem}", output_dir=str(channel_b_dir)
    )
    timings["7. Kanal B (EVREN VLM Video İstek & Yanıt)"] = time.perf_counter() - t0

    # 8. Decision Agent (LLM Çıkarımı)
    t0 = time.perf_counter()
    backend = create_backend(config.vlm, force="server")
    agent = DecisionAgent(
        config=config.decision_agent, vlm_config=config.vlm,
        rag=rag, memory=memory, tools=tools, backend=backend,
    )
    decision_raw = agent.decide(
        event_signals=event_signals,
        scene_graphs=scene_graphs,
        rag_context=rag_context,
        vlm_interpretation=vlm_interpretation,
    )
    timings["8. Karar Ajanı (EVREN LLM Akıl Yürütme)"] = time.perf_counter() - t0

    # 9. Guardrail & Mock Tools
    t0 = time.perf_counter()
    guardrail = OutputGuardrail(config.output.guardrail)
    final_output = guardrail.validate(
        decision_raw["raw_text"],
        decision_raw["retry_fn"],
        rag_risk_level=decision_raw["rag_risk_level"],
    )
    triggered = final_output.get("triggered_mock_tools") or []
    tool_results = [tools.execute(c["tool_name"], c.get("params", {})) for c in triggered]
    timings["9. Guardrail & Mock Tool Çalıştırma"] = time.perf_counter() - t0

    total_time = time.perf_counter() - total_start

    # RAPORLAMA
    print(f"\n{'='*70}")
    print(f"⏱️  PERFORMANS VE ZAMAN DAĞILIMI TABLOSU (Toplam: {total_time:.2f} saniye)")
    print(f"{'='*70}")
    print(f"{'Adım / Bileşen':<45} | {'Süre (sn)':<10} | {'Yüzde (%)':<10}")
    print(f"{'-'*70}")
    for name, dur in timings.items():
        pct = (dur / total_time) * 100
        print(f"{name:<45} | {dur:>8.2f}s  | {pct:>8.1f}%")
    print(f"{'-'*70}")
    print(f"{'TOPLAM İŞLEM SÜRESİ':<45} | {total_time:>8.2f}s  | 100.0%")
    print(f"{'='*70}\n")

    return timings, total_time


if __name__ == "__main__":
    test_video = REPO_ROOT / "videos" / "Normal_Videos_345_x264.mp4"
    if len(sys.argv) > 1:
        test_video = Path(sys.argv[1])
    profile_video(test_video)
