import av
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler

def extract_frames(video_path, num_frames=8):
    """
    Extracts a fixed number of frames from a video file using PyAV.
    """
    print(f"📹 Opening video file: {video_path}")
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    
    # Attempt to get total frames safely
    total_frames = video_stream.frames
    if total_frames is None or total_frames <= 0:
        # If frames count is missing, estimate from duration and frame rate
        duration = video_stream.duration
        time_base = video_stream.time_base
        avg_frame_rate = video_stream.average_rate
        if duration and time_base and avg_frame_rate:
            total_seconds = float(duration * time_base)
            total_frames = int(total_seconds * float(avg_frame_rate))
        else:
            total_frames = 100 # Default fallback
            
    print(f"🎞️  Estimated total frames: {total_frames}")
    
    # Calculate target indices to sample
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    print(f"🎯 Sampling frame indices: {list(indices)}")
    
    frames = []
    current_idx = 0
    sampled_count = 0
    
    # Loop over packet decoding
    for frame in container.decode(video=0):
        if current_idx in indices:
            # Convert PyAV frame to PIL Image
            img = frame.to_image()
            frames.append(img)
            sampled_count += 1
            if sampled_count >= num_frames:
                break
        current_idx += 1
        
    container.close()
    print(f"✅ Extracted {len(frames)} frames.")
    return frames

def img_to_base64_data_url(img):
    """
    Converts a PIL Image into a base64 Data URL.
    """
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

def main():
    model_repo = "mradermacher/TimeLens-7B-i1-GGUF"
    model_file = "TimeLens-7B.i1-Q4_K_M.gguf"
    
    mmproj_repo = "mradermacher/TimeLens-7B-GGUF"
    mmproj_file = "TimeLens-7B.mmproj-f16.gguf"
    
    # Check local models directory first
    local_model = os.path.join("models", model_file)
    local_mmproj = os.path.join("models", mmproj_file)

    if os.path.exists(local_model):
        model_path = local_model
        print(f"✅ Local model found: {model_path}")
    else:
        print("📥 Downloading TimeLens-7B GGUF model...")
        model_path = hf_hub_download(repo_id=model_repo, filename=model_file)
        print(f"✅ Model downloaded to: {model_path}")
    
    if os.path.exists(local_mmproj):
        mmproj_path = local_mmproj
        print(f"✅ Local mmproj found: {mmproj_path}")
    else:
        print("📥 Downloading TimeLens-7B mmproj (vision projector)...")
        mmproj_path = hf_hub_download(repo_id=mmproj_repo, filename=mmproj_file)
        print(f"✅ mmproj downloaded to: {mmproj_path}")
    
    # Initialize the Vulkan-enabled chat handler
    print("🚀 Initializing Vulkan Qwen25VLChatHandler...")
    chat_handler = Qwen25VLChatHandler(clip_model_path=mmproj_path, verbose=True)
    
    # Load the LLM with Vulkan support
    # n_gpu_layers=-1 sends all layers to GPU (sharded over Vulkan devices automatically)
    print("🧠 Loading model into llama.cpp with Vulkan offloading...")
    llm = Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=16384,
        n_gpu_layers=-1,
        verbose=True
    )
    
    # Extract frames from local video
    video_path = "video.mp4"
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Missing local video file: {video_path}")
        
    frames = extract_frames(video_path, num_frames=8)
    
    # Prepare the messages
    content = []
    
    # Add extracted frames as base64 images
    for frame in frames:
        data_url = img_to_base64_data_url(frame)
        content.append({
            "type": "image_url",
            "image_url": {"url": data_url}
        })
        
    # Add text prompt
    prompt_text = (
        "Videoyu analiz et. Videoda ne olduğunu açıkla. "
        "Olayları (events), risk durumunu ve yapılması gereken aksiyonları (actions) tespit et. "
        "SADECE aşağıdaki JSON formatında Türkçe olarak yanıt ver:\n"
        '{"summary":"...","events":[{"time":"00:00","event":"..."}],"risk":"Düşük/Orta/Yüksek","actions":["..."]}'
    )
    content.append({
        "type": "text",
        "text": prompt_text
    })
    
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    print("💬 Generating completion from video frames...")
    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.1,
        max_tokens=400
    )
    
    print("\n--- MODEL ÇIKTISI ---")
    print(response["choices"][0]["message"]["content"])
    print("---------------------\n")

if __name__ == "__main__":
    main()
