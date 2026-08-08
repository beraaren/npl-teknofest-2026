#!/usr/bin/env python3
"""senaryo_3_demo.py için elle çalıştırılan doğrulama testi (v2 — teşhisli).

Çalıştırma:
    cd bera
    ~/.venvs/nlp2026/bin/python test_senaryo_3_demo.py

Ne test eder:
  1. Model dosyaları (LLaVA-v1.6-Mistral-7B Q8_0 + mmproj)
  2. video.mp4'ten 8 kare + grid üretimi
  3. GPU yükleme + VLM çıkarım:
     a) demonun el yapımı CHAT_FORMAT override'ı ile
     b) başarısızsa llama_cpp'nin gömülü llava-1-6 formatı ile
     Ham model çıktısı her durumda ekrana basılır (teşhis için).
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

# HF_TOKEN: .env varsa yükle (demo modülü import anında login çağırıyor)
env_file = REPO / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

PASS, FAIL = "✅ PASS", "❌ FAIL"
results = []


def check(name, fn):
    try:
        out = fn()
        results.append((PASS, name, out or ""))
        print(f"{PASS} {name} {out or ''}")
        return True
    except Exception as e:
        results.append((FAIL, name, str(e)))
        print(f"{FAIL} {name} -> {e}")
        return False


import senaryo_3_demo as demo  # noqa: E402  (login çağrısı .env'den sonra olsun)

PROMPT = (
    "Bu 8 karelik video gridini Türkçe özetle. SADECE JSON döndür: "
    '{"summary": "...", "risk": "Düşük|Orta|Yüksek"}'
)


def t_model_files():
    model_path, mmproj_path = demo.download_model()
    assert Path(model_path).exists(), "model dosyası yok"
    assert Path(mmproj_path).exists(), "mmproj dosyası yok"
    size_gb = Path(model_path).stat().st_size / 1e9
    return f"({size_gb:.1f} GB model + mmproj hazır)"


def t_frames():
    frames = demo.extract_frames(str(REPO / "video.mp4"), num_frames=8)
    assert len(frames) == 8, f"{len(frames)} kare çıktı"
    grid = demo.frames_to_grid(frames)
    assert grid.size == (384 * 4, 216 * 2), f"grid boyutu {grid.size}"
    return f"(8 kare, grid {grid.size[0]}x{grid.size[1]})"


def _infer(llm, data_url):
    resp = llm.create_chat_completion(
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        max_tokens=200,
        temperature=0.1,
    )
    return resp["choices"][0]["message"]["content"]


def _parse_json_loose(text):
    """İlk { ile son } arasını ayrıştır (fence/ön laf toleranslı)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"JSON bloğu yok. Ham çıktı: {text[:300]!r}")
    return json.loads(text[start:end + 1])


def t_inference():
    frames = demo.extract_frames(str(REPO / "video.mp4"), num_frames=8)
    data_url = demo.image_to_data_url(demo.frames_to_grid(frames))

    # a) demonun el yapımı CHAT_FORMAT override'ı
    llm = demo.load_model()
    raw = _infer(llm, data_url)
    print(f"\n--- HAM ÇIKTI (demo override formatı) ---\n{raw!r}\n-----------------------------------------\n")
    try:
        parsed = _parse_json_loose(raw)
        return f"[override formatı çalıştı] risk={parsed.get('risk')}, özet={str(parsed.get('summary'))[:60]}"
    except Exception as e:
        print(f"⚠️ Override formatı JSON üretemedi: {e}")
        print("→ Gömülü llava-1-6 formatıyla yeniden deneniyor...\n")

    # b) gömülü llava-1-6 chat handler (override YOK)
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava16ChatHandler
    model_path, mmproj_path = demo.download_model()
    chat_handler = Llava16ChatHandler(clip_model_path=mmproj_path, verbose=False)
    llm2 = Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_gpu_layers=-1,
        n_ctx=10384,
        split_mode=1,
        main_gpu=0,
        tensor_split=[1, 0],
        verbose=False,
    )
    raw2 = _infer(llm2, data_url)
    print(f"\n--- HAM ÇIKTI (gömülü llava-1-6) ---\n{raw2!r}\n------------------------------------\n")
    parsed = _parse_json_loose(raw2)
    assert "summary" in parsed and "risk" in parsed, f"eksik alanlar: {list(parsed)}"
    return f"[gömülü format çalıştı] risk={parsed['risk']}, özet={str(parsed['summary'])[:60]}"


if __name__ == "__main__":
    print("=" * 60)
    print("senaryo_3_demo doğrulama testi v2 (venv: ~/.venvs/nlp2026)")
    print("=" * 60)
    ok = True
    ok &= check("1. Model dosyaları indirildi", t_model_files)
    ok &= check("2. Kare çıkarma + grid", t_frames)
    ok &= check("3. GPU yükleme + VLM çıkarım + JSON", t_inference)
    print("=" * 60)
    passed = sum(1 for r in results if r[0] == PASS)
    print(f"SONUÇ: {passed}/{len(results)} test geçti")
    sys.exit(0 if ok else 1)
