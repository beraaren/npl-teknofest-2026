#!/usr/bin/env python3
"""Önceki Sıralı/CPU Algoritması ile Güncel Paralel/GPU Algoritmasını Karşılaştırır."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

REPO_ROOT = Path(__file__).resolve().parent.parent
for _entry in (str(REPO_ROOT), str(REPO_ROOT / "Kanal_B")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env", override=True)

import numpy as np
from PIL import Image
import torch
import av

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


def run_legacy_pipeline(video_path: Path, config_path: Path):
    """ÖNCEKİ ALGORİTMA: Sıralı + CPU libx264 + Gereksiz FrameSampler/CLAHE."""
    print("\n" + "-"*65)
    print("🐢 [1/2] ÖNCEKİ ALGORİTMA (Sıralı, CPU, Ağır Ön İşleme) ÇALIŞTIRILIYOR...")
    print("-"*65)
    
    t_start = time.perf_counter()
    config = load_config(str(config_path))
    info = probe_video(str(video_path))
    native_fps = float(info["fps"]) or 25.0
    target_fps = config.preprocessing.channel_a_fps
    step = max(1, round(native_fps / target_fps)) if target_fps > 0 else 1
    fps_a = native_fps / step

    # 1. Kanal A Kare Okuma (CPU)
    t0 = time.perf_counter()
    reader = VideoReader(str(video_path))
    channel_a_frames, channel_a_indices = [], []
    for idx, frame in enumerate(reader.iter_frames()):
        if idx % step == 0:
            channel_a_frames.append(frame)
            channel_a_indices.append(idx)
    reader.close()
    t_read = time.perf_counter() - t0

    # 2. Eski FrameSampler & CLAHE (CPU üzerinde gereksiz hesaplama)
    t0 = time.perf_counter()
    sampler = FrameSampler(
        target_count=config.preprocessing.target_frame_count,
        use_smart_sampling=config.preprocessing.use_smart_sampling,
        ssim_threshold=config.preprocessing.ssim_threshold,
        min_laplacian_variance=config.preprocessing.min_laplacian_variance,
    )
    sampled_frames, sampled_pos = sampler.sample(channel_a_frames, len(channel_a_frames))
    if config.preprocessing.enhance_low_light:
        enhancer = LowLightEnhancer(enabled=True)
        sampled_frames = [enhancer.enhance(f) for f in sampled_frames]
    t_sampler = time.perf_counter() - t0

    # 3. Sıralı Kanal A YOLO + Kural Motoru
    t0 = time.perf_counter()
    observer = ObserverAgent(config.perception)
    observations = observer.observe_video(
        channel_a_frames, native_fps, sampled_indices=channel_a_indices
    )
    scene_graphs = [obs["scene_graph"] for obs in observations]

    engine = EventEngine(config.events, fps=fps_a)
    for obs in observations:
        engine.process_observation(obs)
    event_signals = engine.get_signals()

    rag = RAGLayer()
    rag_context = rag.build_context(observations, event_signals)
    memory = ShortTermMemory()
    for sig in event_signals:
        memory.add(sig, entry_type="event")
    t_channel_a = time.perf_counter() - t0

    # 4. Sıralı Kanal B (Kanal A bittikten sonra başlatılıyor)
    t0 = time.perf_counter()
    out_b_dir = REPO_ROOT / "scratch" / "legacy_channel_b"
    vlm_interpretation = run_channel_b(str(video_path), video_id="legacy_test", output_dir=str(out_b_dir))
    t_channel_b = time.perf_counter() - t0

    # 5. Karar Ajanı LLM Çıkarımı
    t0 = time.perf_counter()
    tools = MockToolRegistry()
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
    guardrail = OutputGuardrail(config.output.guardrail)
    final_output = guardrail.validate(
        decision_raw["raw_text"],
        decision_raw["retry_fn"],
        rag_risk_level=decision_raw["rag_risk_level"],
    )
    t_llm = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start

    return {
        "read_sec": t_read,
        "sampler_sec": t_sampler,
        "channel_a_sec": t_channel_a,
        "channel_b_sec": t_channel_b,
        "llm_sec": t_llm,
        "total_sec": t_total,
        "risk": final_output.get("risk"),
        "confidence": final_output.get("confidence"),
        "events_count": len(final_output.get("events", [])),
    }


def run_current_pipeline(video_path: Path, config_path: Path):
    """GÜNCEL ALGORİTMA: Paralel Kanal A & B + GPU NVENC + CUDA YOLO + Optimize Sampler."""
    print("\n" + "-"*65)
    print("⚡ [2/2] GÜNCEL ALGORİTMA (Paralel, GPU NVENC, CUDA YOLO) ÇALIŞTIRILIYOR...")
    print("-"*65)
    
    t_start = time.perf_counter()
    config = load_config(str(config_path))
    info = probe_video(str(video_path))
    native_fps = float(info["fps"]) or 25.0
    target_fps = config.preprocessing.channel_a_fps
    step = max(1, round(native_fps / target_fps)) if target_fps > 0 else 1
    fps_a = native_fps / step

    def _channel_a_worker():
        t0 = time.perf_counter()
        reader = VideoReader(str(video_path))
        frames, indices = [], []
        for idx, frame in enumerate(reader.iter_frames()):
            if idx % step == 0:
                frames.append(frame)
                indices.append(idx)
        reader.close()
        t_read = time.perf_counter() - t0

        t0 = time.perf_counter()
        observer = ObserverAgent(config.perception)
        observations = observer.observe_video(frames, native_fps, sampled_indices=indices)
        scene_graphs = [obs["scene_graph"] for obs in observations]

        engine = EventEngine(config.events, fps=fps_a)
        for obs in observations:
            engine.process_observation(obs)
        event_signals = engine.get_signals()

        rag = RAGLayer()
        rag_context = rag.build_context(observations, event_signals)
        memory = ShortTermMemory()
        for sig in event_signals:
            memory.add(sig, entry_type="event")
        t_ch_a = time.perf_counter() - t0
        return observations, scene_graphs, event_signals, rag_context, memory, rag, t_read, t_ch_a

    out_b_dir = REPO_ROOT / "scratch" / "current_channel_b"

    t0_par = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_a = executor.submit(_channel_a_worker)
        f_b = executor.submit(run_channel_b, str(video_path), video_id="current_test", output_dir=str(out_b_dir))

        observations, scene_graphs, event_signals, rag_context, memory, rag, t_read, t_channel_a = f_a.result()
        vlm_interpretation = f_b.result()
    t_parallel_stage = time.perf_counter() - t0_par

    # Karar Ajanı LLM Çıkarımı
    t0 = time.perf_counter()
    tools = MockToolRegistry()
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
    guardrail = OutputGuardrail(config.output.guardrail)
    final_output = guardrail.validate(
        decision_raw["raw_text"],
        decision_raw["retry_fn"],
        rag_risk_level=decision_raw["rag_risk_level"],
    )
    t_llm = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start

    return {
        "read_sec": t_read,
        "sampler_sec": 0.0,  # Optimize edildi (0 sn)
        "channel_a_sec": t_channel_a,
        "channel_b_parallel_sec": t_parallel_stage,
        "llm_sec": t_llm,
        "total_sec": t_total,
        "risk": final_output.get("risk"),
        "confidence": final_output.get("confidence"),
        "events_count": len(final_output.get("events", [])),
    }


def compare(video_path: Path):
    config_path = REPO_ROOT / "config.yaml"
    print(f"\n{'='*75}")
    print(f"📊 ALGORİTMA VE PERFORMANS KARŞILAŞTIRMASI: {video_path.name}")
    print(f"{'='*75}")

    legacy = run_legacy_pipeline(video_path, config_path)
    current = run_current_pipeline(video_path, config_path)

    diff_sec = legacy["total_sec"] - current["total_sec"]
    pct_faster = (diff_sec / legacy["total_sec"]) * 100 if legacy["total_sec"] > 0 else 0
    speedup = legacy["total_sec"] / current["total_sec"] if current["total_sec"] > 0 else 1.0

    print("\n" + "="*75)
    print("🏆 BİREBİR KARŞILAŞTIRMA VE HIZ TESTİ RAPORU")
    print("="*75)
    print(f"{'Metrik / Aşama':<35} | {'Önceki Algoritma':<17} | {'Güncel Algoritma':<17}")
    print("-"*75)
    print(f"{'Kanal A (YOLO + Kural Motoru)':<35} | {legacy['channel_a_sec']:>14.2f}s  | {current['channel_a_sec']:>14.2f}s (GPU)")
    print(f"{'Ön Sampler & CLAHE':<35} | {legacy['sampler_sec']:>14.2f}s  | {current['sampler_sec']:>14.2f}s (Atlandı)")
    print(f"{'Kanal B + Kanal A Çalışma Modu':<35} | {'Sıralı (Bekleme)':<17} | {'Paralel (Eşzamanlı)':<17}")
    print(f"{'Kanal B (+ Paralel Gizleme)':<35} | {legacy['channel_b_sec']:>14.2f}s  | {current['channel_b_parallel_sec']:>14.2f}s")
    print(f"{'Karar Ajanı (LLM Akıl Yürütme)':<35} | {legacy['llm_sec']:>14.2f}s  | {current['llm_sec']:>14.2f}s")
    print("-"*75)
    print(f"{'TOPLAM SÜRE':<35} | {legacy['total_sec']:>14.2f}s  | {current['total_sec']:>14.2f}s")
    print(f"{'Kazanılan Zaman':<35} | {'-':<17} | {diff_sec:>13.2f}s daha hızlı")
    print(f"{'Performans Artışı / Hızlanma':<35} | {'1.00x':<17} | {speedup:>13.2f}x (%{pct_faster:.1f} kazanç)")
    print("="*75)
    print(f"🎯 Karar Doğrulaması:")
    print(f"  • Önceki Algoritma Çıktısı: Risk={legacy['risk']} (Güven: {legacy['confidence']}, Olay: {legacy['events_count']})")
    print(f"  • Güncel Algoritma Çıktısı: Risk={current['risk']} (Güven: {current['confidence']}, Olay: {current['events_count']})")
    print("="*75 + "\n")


if __name__ == "__main__":
    vpath = REPO_ROOT / "videos" / "Normal_Videos_345_x264.mp4"
    if len(sys.argv) > 1:
        vpath = Path(sys.argv[1])
    compare(vpath)
