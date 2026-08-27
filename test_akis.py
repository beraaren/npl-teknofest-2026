#!/usr/bin/env python3
"""Uçtan uca akışı konsolda adım adım gösteren test scripti.

src/main.py ile BİREBİR aynı adımları çalıştırır; farkı her aşamanın
girdi/çıktısını konsola basmasıdır. VLM sunucusunun ayakta olması gerekir:

    GGML_VULKAN_DEVICE=0 ~/.venvs/nlp2026/bin/python run_vlm_server.py ... (ayakta)

Çalıştırma:
    cd bera
    ~/.venvs/nlp2026/bin/python test_akis.py                 # video.mp4, server backend
    ~/.venvs/nlp2026/bin/python test_akis.py --video baska.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def hdr(title: str) -> None:
    print(f"\n{'=' * 70}\n■ {title}\n{'=' * 70}")


def sub(title: str) -> None:
    print(f"\n--- {title} ---")


def pj(obj, limit: int = 0) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if limit and len(text) > limit:
        text = text[:limit] + f"\n... ({len(text)} karakter, kısaltıldı)"
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="Video dosyası (verilmezse --random-dir'den rastgele seçilir)")
    ap.add_argument(
        "--random-dir",
        default=str(REPO.parent / "INDIR_BENCHMARK" / "videos"),
        help="Rastgele video seçilecek dizin (alt dizinler = kategoriler)",
    )
    ap.add_argument("--category", default=None, help="Sadece bu kategoriden seç (örn. Arson)")
    ap.add_argument("--seed", type=int, default=None, help="Tekrarlanabilir seçim için sabit tohum")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--backend", default="server")
    ap.add_argument("--detector", default="ultralytics")
    args = ap.parse_args()

    if args.video is None:
        import random
        root = Path(args.random_dir)
        if args.category:
            candidates = sorted((root / args.category).glob("*.mp4"))
        else:
            candidates = sorted(p for p in root.rglob("*.mp4"))
        if not candidates:
            raise SystemExit(f"Video bulunamadı: {root} (category={args.category})")
        rng = random.Random(args.seed)
        args.video = str(rng.choice(candidates))
    print(f"🎲 Seçilen video: {args.video}")

    if args.backend == "server":
        # Sağlık kontrolü yapılandırmadan okunur; sabit bir yerel adrese
        # bakılmaz. Sağlayıcı artık TEKNOFEST EVREN (uzak, TLS, bearer token)
        # olduğu için eski 127.0.0.1:8080 denetimi her koşulda başarısızdı.
        import urllib.error
        import urllib.request

        from src.config import load_config as _load_config

        _cfg = _load_config(args.config)
        _srv = _cfg.vlm.server
        _key = os.environ.get(_srv.api_key_env, "").strip()
        _req = urllib.request.Request(f"{_srv.base_url.rstrip('/')}/models")
        if _key:
            _req.add_header("Authorization", f"Bearer {_key}")
        try:
            with urllib.request.urlopen(_req, timeout=10) as _resp:
                _models = [m.get("id") for m in json.load(_resp).get("data", [])]
            print(f"✅ Çıkarım servisi erişilebilir: {_srv.base_url}")
            print(f"   modeller: {', '.join(m for m in _models if m)}")
            for _name, _label in ((_srv.model_name, "görüntü/metin"), (_srv.video_model, "video")):
                if _name not in _models:
                    print(
                        f"   ⚠ '{_name}' ({_label}) sunucu listesinde yok; "
                        f"istek sessizce başka bir modele yönlendirilebilir."
                    )
        except urllib.error.HTTPError as _e:
            raise SystemExit(
                f"❌ Çıkarım servisi HTTP {_e.code} döndürdü ({_srv.base_url}).\n"
                f"   '{_srv.api_key_env}' anahtarı geçerli mi? (.env dosyasını kontrol et)"
            )
        except Exception as _e:
            raise SystemExit(
                f"❌ Çıkarım servisine ulaşılamadı: {_srv.base_url}\n"
                f"   Hata: {_e}\n"
                f"   İnternet bağlantısını ve '{_srv.api_key_env}' değerini kontrol et."
            )

    import numpy as np
    from PIL import Image

    from src.config import load_config
    from src.events.event_engine import EventEngine
    from src.models.vlm_backend import create_backend
    from src.output.guardrail import OutputGuardrail
    from src.perception.observer_agent import ObserverAgent
    from src.preprocessing.enhancer import LowLightEnhancer
    from src.preprocessing.frame_sampler import FrameSampler
    from src.preprocessing.video_reader import VideoReader
    from src.reasoning.decision_agent import DecisionAgent
    from src.reasoning.memory import ShortTermMemory
    from src.reasoning.mock_tools import MockToolRegistry
    from src.reasoning.rag_layer import RAGLayer

    config = load_config(args.config)
    config.perception.detector_backend = args.detector

    # ---------------------------------------------------------------- AŞAMA 1
    hdr("AŞAMA 1 — Video okuma (Kanal A yoğun örnekleme)")
    reader = VideoReader(args.video)
    native_fps = reader.fps
    target_fps = config.preprocessing.channel_a_fps
    step = max(1, round(native_fps / target_fps)) if target_fps > 0 else 1
    fps_a = native_fps / step
    channel_a_frames, channel_a_indices = [], []
    for i, frame in enumerate(reader.iter_frames()):
        if i % step == 0:
            channel_a_frames.append(frame)
            channel_a_indices.append(i)
    reader.close()
    print(f"Video: {args.video} | {reader.total_frames} kare | {native_fps:.2f} fps")
    print(f"Kanal A: her {step}. kare → {len(channel_a_frames)} kare @ {fps_a:.1f} fps")

    # ---------------------------------------------------------------- AŞAMA 2
    hdr("AŞAMA 2 — Kanal B akıllı örnekleme + ön işleme")
    sampler = FrameSampler(
        target_count=config.preprocessing.target_frame_count,
        use_smart_sampling=config.preprocessing.use_smart_sampling,
        ssim_threshold=config.preprocessing.ssim_threshold,
        min_laplacian_variance=config.preprocessing.min_laplacian_variance,
    )
    sampled_frames, sampled_pos = sampler.sample(channel_a_frames, len(channel_a_frames))
    sampled_indices = [channel_a_indices[p] for p in sampled_pos]
    print(f"Seçilen {len(sampled_frames)} kare (gerçek video indeksleri): {sampled_indices}")
    if config.preprocessing.enhance_low_light:
        enhancer = LowLightEnhancer(
            enabled=True,
            clip_limit=config.preprocessing.clahe_clip_limit,
            grid_size=tuple(config.preprocessing.clahe_grid_size),
        )
        sampled_frames = [enhancer.enhance(f) for f in sampled_frames]
        print("CLAHE düşük ışık iyileştirmesi uygulandı.")
    target_size = (config.preprocessing.frame_width, config.preprocessing.frame_height)
    sampled_frames = [np.array(Image.fromarray(f).resize(target_size)) for f in sampled_frames]
    print(f"Yeniden boyutlandırma: {target_size[0]}x{target_size[1]}")

    # ---------------------------------------------------------------- AŞAMA 3
    hdr("AŞAMA 3 — Gözlemci Ajan (YOLO + tracker, Kanal A tamamı)")
    observer = ObserverAgent(config.perception)
    observations = observer.observe_video(channel_a_frames, native_fps, sampled_indices=channel_a_indices)
    scene_graphs = [obs["scene_graph"] for obs in observations]
    nonempty = [(o.get("frame_idx", i), o["scene_graph"].get("nodes", [])) for i, o in enumerate(observations)]
    det = [(fi, nodes) for fi, nodes in nonempty if nodes]
    print(f"{len(observations)} kare gözlendi, {len(det)} karede nesne var.")
    from collections import Counter
    counts = Counter(n.get("class", "?") for _, nodes in det for n in nodes)
    print(f"Sınıf dağılımı (tüm kareler): {dict(counts)}")
    if det:
        sub(f"Örnek sahne grafi (kare {det[-1][0]})")
        pj({"frame_idx": det[-1][0], "nodes": det[-1][1][:6]}, limit=1200)

    # ---------------------------------------------------------------- AŞAMA 4
    hdr("AŞAMA 4 — Olay Tespit Motoru (geometrik kurallar)")
    event_engine = EventEngine(config.events, fps=fps_a)
    for obs in observations:
        event_engine.process_observation(obs)
    event_signals = event_engine.get_signals()
    print(f"{len(event_signals)} sinyal:")
    for s in event_signals:
        print(f"  [{s.get('timestamp')}] {s.get('event_type')} (güven {s.get('confidence')}): {s.get('description')}")

    # ---------------------------------------------------------------- AŞAMA 5
    hdr("AŞAMA 5 — RAG Katmanı (TF-IDF vektör arama, risk kataloğu)")
    rag = RAGLayer()
    rag_context = rag.build_context(observations, event_signals)
    pj(rag_context)

    # ---------------------------------------------------------------- AŞAMA 6
    hdr("AŞAMA 6 — Hafıza + Araç kataloğu")
    memory = ShortTermMemory()
    for sig in event_signals:
        memory.add(sig, entry_type="event")
    tools = MockToolRegistry()
    print(memory.to_prompt_context())
    print(f"\nAraçlar: {', '.join(tools.tools.keys())}")

    # ---------------------------------------------------------------- AŞAMA 7
    hdr("AŞAMA 7 — Kanal B (S8): bağımsız VLM yorumu")
    backend = create_backend(config.vlm, force=args.backend)
    print(f"VLM backend: {backend.name()}  ({config.vlm.server.base_url})")
    agent = DecisionAgent(
        config=config.decision_agent, vlm_config=config.vlm,
        rag=rag, memory=memory, tools=tools, backend=backend,
    )
    kanal_b_dir = str(REPO / "Kanal_B")
    if kanal_b_dir not in sys.path:
        sys.path.insert(0, kanal_b_dir)
    from pipeline import run_channel_b  # Kanal_B/pipeline.py

    vlm_interpretation = run_channel_b(
        args.video, video_id=Path(args.video).stem,
        output_dir=str(REPO / "outputs" / "channel_b"),
    )
    print("Kanal_B paketi kullanıldı (run_channel_b).")
    sub("S8 — ham model çıktısı")
    print(vlm_interpretation.get("raw_model_output", "(yok)")[:1500])
    sub("S8 — ayrıştırılmış yorum")
    pj({k: v for k, v in vlm_interpretation.items() if k != "raw_model_output"}, limit=1200)

    # ---------------------------------------------------------------- AŞAMA 8
    hdr("AŞAMA 8 — Karar Ajanı (VLM, düşünce zinciri)")
    prompt = agent._build_prompt(event_signals, scene_graphs, rag_context, vlm_interpretation)
    sub(f"Karar promptu ({len(prompt)} karakter)")
    pj(prompt, limit=2500)
    decision_raw = agent.decide(
        event_signals=event_signals, scene_graphs=scene_graphs,
        rag_context=rag_context, vlm_interpretation=vlm_interpretation,
    )
    sub("Karar ajanı — ham çıktı")
    print(decision_raw["raw_text"])

    # ---------------------------------------------------------------- AŞAMA 9
    hdr("AŞAMA 9 — Guardrail (şema doğrulama + retry)")
    guardrail = OutputGuardrail(config.output.guardrail)
    final_output = guardrail.validate(
        decision_raw["raw_text"], decision_raw["retry_fn"],
        rag_risk_level=decision_raw["rag_risk_level"],
    )
    pj(final_output)

    # ---------------------------------------------------------------- AŞAMA 10
    hdr("AŞAMA 10 — Araç çalıştırma (mock)")
    triggered = final_output.get("triggered_mock_tools", [])
    if not triggered:
        suggested = tools.suggest_tools(
            final_output["risk"], [e.get("event_type", "") for e in final_output.get("events", [])]
        )
        triggered = [
            {"tool_name": s["tool_name"], "params": {"location": "saha", "reason": final_output["summary"][:100]}}
            for s in suggested
        ]
        final_output["triggered_mock_tools"] = triggered
        print("Model araç seçmedi; kural tabanlı öneri kullanıldı.")
    for call in triggered:
        result = tools.execute(call["tool_name"], call["params"])
        print(f"  {call['tool_name']} {call.get('params')} → {result['status']}: {result.get('mock_result', result.get('message', ''))}")

    hdr("SONUÇ")
    print(f"Risk: {final_output['risk']} | Güven: {final_output['confidence']}")
    print(f"Özet: {final_output['summary']}")
    print(f"Aksiyonlar: {final_output['actions']}")


if __name__ == "__main__":
    main()
