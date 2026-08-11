#!/usr/bin/env bash
# VLM sunucusunu başlatır: LLaVA-1.6-Mistral-7B (Q8_0) + Vulkan, tek GPU (RX 9070).
#
# Kanıtlanmış stabil ayarlar (senaryo_3_demo ile aynı):
#   - Tüm katmanlar GPU0'da (tensor_split 1 0 → RTX 4060'a BÖLME YOK)
#   - n_ctx 32768 (modelin eğitim bağlamı; KV ≈ 4 GiB, ~1 GB VRAM marjı)
#   - run_vlm_server.py: gömülü şablon [INST] üretmediği için
#     LLaVA-1.6 [INST] + system rolü destekli şablonu uygular
#
# Kullanım:
#   ./start_vlm_server.sh          # ön planda (logları canlı izle)
#   ./start_vlm_server.sh --bg     # arka planda (log: logs/vlm_server.log)
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="$HOME/.venvs/nlp2026/bin/python"
SNAP="$HOME/.cache/huggingface/hub/models--cjpais--llava-1.6-mistral-7b-gguf/snapshots/6019df415777605a8364e2668aa08b7e354bf0ba"
MODEL="$SNAP/llava-v1.6-mistral-7b.Q8_0.gguf"
MMPROJ="$SNAP/mmproj-model-f16.gguf"
PORT=8080

if curl -s "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"id"'; then
    echo "Sunucu zaten ayakta: http://127.0.0.1:$PORT"
    exit 0
fi

CMD=(
    "$PYTHON" run_vlm_server.py
    --model "$MODEL"
    --clip_model_path "$MMPROJ"
    --chat_format llava-1-6
    --host 127.0.0.1 --port "$PORT"
    --n_ctx 32768
    --n_gpu_layers -1
    --split_mode 1 --main_gpu 0 --tensor_split 1 0
)

if [[ "${1:-}" == "--bg" ]]; then
    mkdir -p logs
    nohup "${CMD[@]}" > logs/vlm_server.log 2>&1 &
    echo "Sunucu arka planda başlatıldı (PID $!), log: logs/vlm_server.log"
    echo -n "Hazır olması bekleniyor"
    for _ in $(seq 1 36); do
        if curl -s "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"id"'; then
            echo " → HAZIR: http://127.0.0.1:$PORT"
            exit 0
        fi
        echo -n "."
        sleep 5
    done
    echo
    echo "3 dakikada hazır olmadı; loga bak: logs/vlm_server.log" >&2
    exit 1
else
    exec "${CMD[@]}"
fi
