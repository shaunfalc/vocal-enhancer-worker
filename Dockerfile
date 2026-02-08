# VocalEnhancer worker: Resemble Enhance on RunPod (always-on pod).
# Build from repo root: docker build -t vocal-enhancer-worker .
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA first (match CUDA 12.1 in base)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy and install remaining deps (don't reinstall torch — use requirements-docker.txt)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Resemble Enhance from Git with --no-deps to avoid deepspeed (training-only) which fails in Docker
RUN pip install --no-cache-dir "git+https://github.com/resemble-ai/resemble-enhance.git@main" --no-deps

COPY app.py .

# Expose port for RunPod HTTP proxy (Expose HTTP Ports: 8000)
EXPOSE 8000

# Bind all interfaces so RunPod proxy can reach us
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
