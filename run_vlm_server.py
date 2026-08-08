#!/usr/bin/env python3
"""VLM sunucusu: llama_cpp.server + LLaVA-1.6 Mistral [INST] şablon düzeltmesi.

Neden bu wrapper var:
  llama-cpp-python 0.3.34'ün gömülü Llava16ChatHandler.CHAT_FORMAT'ı [INST]
  üretmiyor (jenerik bir şablon) ve model bundan dolayı prompt'u papağan gibi
  tekrarlıyor. `--chat_format mtmd` ise GGUF'taki Mistral şablonunu kullanıyor
  ama o şablon "system" rolünde exception fırlatıyor (Kanal_B system gönderiyor)
  ve liste-tipi content'i birleştiremiyor.

  Bu wrapper, senaryo_3_demo.py'de kanıtlanmış [INST] şablonunu (system rolü
  desteği ekli) sınıfa uygulayıp normal sunucuyu başlatır. Kullanımı
  `python -m llama_cpp.server` ile aynı:

  GGML_VULKAN_DEVICE=0 ~/.venvs/nlp2026/bin/python run_vlm_server.py \
      --model <model.gguf> --clip_model_path <mmproj.gguf> \
      --chat_format llava-1-6 --host 127.0.0.1 --port 8080 \
      --n_ctx 8192 --n_gpu_layers -1 --split_mode 1 --main_gpu 0
"""
import llama_cpp.llama_chat_format as lcf

# Demo ile aynı [INST] akışı + system mesajı prompt başına düz metin eklenir.
LLAVA16_INST_FORMAT = (
    "{% for message in messages %}"
    "{% if message.role == 'system' %}"
    "{{ message.content + '\n\n' }}"
    "{% elif message.role == 'user' %}"
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

lcf.Llava16ChatHandler.CHAT_FORMAT = LLAVA16_INST_FORMAT

from llama_cpp.server.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
