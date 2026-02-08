# VocalEnhancer worker: Resemble Enhance on RunPod (always-on pod).
# Build from repo root: docker build -t vocal-enhancer-worker .
# Use 12.1.1 (supported); 12.1.0 is deprecated per NVIDIA container support policy.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv \
    libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA first (match CUDA 12.1 in base)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy and install remaining deps (don't reinstall torch — use requirements-docker.txt)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Resemble Enhance: clone and copy package, then patch train.py so deepspeed is optional (inference-only).
# Patch is in same RUN as clone so cache never yields an unpatched tree.
COPY patch_deepspeed_optional.py .
RUN git clone --depth 1 https://github.com/resemble-ai/resemble-enhance.git /tmp/resemble-enhance \
    && cp -r /tmp/resemble-enhance/resemble_enhance /app/ \
    && cp -r /tmp/resemble-enhance/config /app/ \
    && rm -rf /tmp/resemble-enhance \
    && python3 /app/patch_deepspeed_optional.py

COPY app.py .

# Expose port for RunPod HTTP proxy (Expose HTTP Ports: 8000)
EXPOSE 8000

# Bind all interfaces so RunPod proxy can reach us
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
