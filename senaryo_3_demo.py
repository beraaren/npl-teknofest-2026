#!/usr/bin/env python3
"""
TEKNOFEST 2026 - Senaryo 3: Video Analiz ve Karar Destek
llama.cpp + Vulkan backend — 2 GPU (AMD RX 9070 16GB + NVIDIA RTX 4060 8GB)
PyTorch KULLANILMIYOR.
"""
import av
import numpy as np
import json
import base64
import io
import os
import subprocess
from PIL import Image
from huggingface_hub import hf_hub_download, login

login(token=os.environ.get("HF_TOKEN"))


def get_vulkan_gpus():
    """Vulkan GPU'ları tespit et"""
    try:
        out = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=10)
        lines = out.stdout.split("\n")
        gpus = []
        for i, line in enumerate(lines):
            if "deviceName" in line:
                name = line.split("=")[1].strip()
                gpus.append(name)
        return gpus
    except Exception:
        return []


def download_model():
    """LLaVA-v1.6 Mistral 7B GGUF + mmproj indir"""
    print("📥 Model dosyaları indiriliyor...")

    model_path = hf_hub_download(
        repo_id="cjpais/llava-1.6-mistral-7b-gguf",
        filename="llava-v1.6-mistral-7b.Q8_0.gguf",
    )
    print(f"  ✅ Model: {model_path}")

    mmproj_path = hf_hub_download(
        repo_id="cjpais/llava-1.6-mistral-7b-gguf",
        filename="mmproj-model-f16.gguf",
    )
    print(f"  ✅ mmproj: {mmproj_path}")

    return model_path, mmproj_path


def load_model():
    """Modeli Vulkan backend ile 2 GPU'ya yükle"""
    model_path, mmproj_path = download_model()

    gpus = get_vulkan_gpus()
    print(f"\n{'='*60}")
    print(f"🖥️  Vulkan GPU'lar:")
    for i, name in enumerate(gpus):
        print(f"  GPU{i}: {name}")
    print(f"{'='*60}\n")

    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava16ChatHandler

    chat_handler = Llava16ChatHandler(clip_model_path=mmproj_path, verbose=False)
    # LLaVA-1.6 Mistral modeli [INST] formatı kullanıyor, varsayılan vicuna değil
    chat_handler.CHAT_FORMAT = (
        "{% for message in messages %}"
        "{% if message.role == 'user' %}"
        "[INST] {% if message.content is string %}{{ message.content }}{% else %}"
        "{% for content in message.content %}"
        "{% if content.type == 'image_url' and content.image_url is string %}{{ content.image_url }}{% endif %}"
        "{% if content.type == 'image_url' and content.image_url is mapping %}{{ content.image_url.url }}{% endif %}"
        "{% if content.type == 'text' %}\n{{ content.text }}{% endif %}"
        "{% endfor %}"
        "{% endif %} [/INST]"
        "{% elif message.role == 'assistant' %}"
        "{{ message.content }}"
        "{% endif %}"
        "{% endfor %}"
    )

    # Her iki GPU'yu Vulkan ile kullan
    # GPU0 = AMD RX 9070 (16GB), GPU1 = NVIDIA RTX 4060 (8GB)
    # tensor_split oranı: 16GB / 8GB → 0.67 / 0.33
    llm = Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_gpu_layers=-1,       # tüm katmanları GPU'ya
        n_ctx=10384,
        split_mode=1,          # LLAMA_SPLIT_MODE_LAYER
        main_gpu=0,            # AMD RX 9070 birincil
        tensor_split=[1, 0],  # VRAM oranı
        verbose=False,
    )

    print("✅ Model 2 GPU'ya Vulkan ile yüklendi!")
    return llm


def extract_frames(video_path, num_frames=8):
    """Videodan eşit aralıklı kareler çıkar"""
    container = av.open(video_path)
    total = container.streams.video[0].frames
    # Bazı videolarda frames metadata boş olabilir, güvenli fallback
    if total is None or total <= 0:
        duration = container.streams.video[0].duration
        time_base = container.streams.video[0].time_base
        avg_rate = container.streams.video[0].average_rate
        if duration and time_base and avg_rate:
            total = int(float(duration * time_base) * float(avg_rate))
        else:
            # Son çare: tüm kareleri say
            total = sum(1 for _ in container.decode(video=0))
            container.seek(0)
    indices = set(np.linspace(0, total - 1, num_frames, dtype=int).tolist())

    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i > max(indices):
            break
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def frames_to_grid(frames, cols=4):
    """Kareleri tek bir grid görüntüsüne birleştir"""
    rows = (len(frames) + cols - 1) // cols
    tw, th = 384, 216

    grid = Image.new("RGB", (tw * cols, th * rows))
    for idx, f in enumerate(frames):
        img = Image.fromarray(f).resize((tw, th))
        r, c = divmod(idx, cols)
        grid.paste(img, (c * tw, r * th))
    return grid


def image_to_data_url(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def main():
    llm = load_model()

    system_prompt = """Analyze this image grid showing 8 video frames (4 columns x 2 rows, left-to-right, top-to-bottom in time order).
Describe what is happening in the video. Identify any events, risks, and recommended actions.
Respond ONLY in valid JSON format in Turkish language:
{"summary":"...","events":[{"time":"00:00","event":"..."}],"risk":"Düşük/Orta/Yüksek","actions":["..."]}"""

    print("📹 Video dosyası açılıyor...")
    try:
        # Önce yerel videoyu dene
        video_path = "video.mp4"
        av.open(video_path)  # Test edelim, bozuksa Hata verir
    except Exception as e:
        print(f"⚠️ Yerel video.mp4 açılamadı ({e}). Yedek demo video indiriliyor...")
        video_path = hf_hub_download(
            repo_id="raushan-testing-hf/videos-test",
            filename="sample_demo_1.mp4",
            repo_type="dataset",
        )

    print("🎞️  Video kareler çıkarılıyor...")
    frames = extract_frames(video_path, num_frames=8)
    grid = frames_to_grid(frames)
    data_url = image_to_data_url(grid)

    print("🧠 Vulkan üzerinden çıkarım (inference) yapılıyor...")
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": system_prompt},
                ],
            }
        ],
        max_tokens=600,
        temperature=0.1,
        repeat_penalty=1.2,
        top_p=0.9,
    )

    result_text = response["choices"][0]["message"]["content"]

    print("\n--- MODEL ÇIKTISI ---")
    print(result_text)
    print("---------------------\n")

    try:
        # Model ```json fence'i veya kısa bir ön laf ekleyebilir:
        # ilk { ile son } arasını parse et (decision_agent._extract_json ile aynı mantık)
        clean = result_text.replace("```json", "").replace("```", "")
        start, end = clean.find("{"), clean.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("çıktıda JSON bloğu yok")
        parsed = json.loads(clean[start:end + 1])
        print("✅ Başarılı: Çıktı JSON formatına uygun!")
        print(f"📌 Özet: {parsed.get('summary')}")
        print(f"⚠️  Risk: {parsed.get('risk')}")
        print(f"🛠  Aksiyonlar: {', '.join(parsed.get('actions', []))}")
    except json.JSONDecodeError:
        print("❌ Hata: JSON formatı üretilemedi.")


if __name__ == "__main__":
    main()
