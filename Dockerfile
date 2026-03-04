# VocalEnhancer worker: Resemble Enhance on RunPod Serverless.
# Build from repo root: docker build -t vocal-enhancer-worker .
#
# Uses RunPod's official PyTorch 2.4 base (cuda12.4.1) — covers Ada Lovelace (RTX 4090, ADA_32_PRO) + RTX 3090/A5000.
# Fixes CUDA kernel mismatch error on Ada Lovelace GPUs (sm_89) that affected cuda12.1.1 builds.
#
# RunPod Serverless setup:
#   1. Push image to GHCR (GitHub Actions CI handles this)
#   2. Create a Serverless Endpoint in RunPod dashboard, point to this image
#   3. Set env vars: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
#   4. (Optional) Mount a network volume at /runpod-volume and set TORCH_HOME=/runpod-volume/torch_cache
#      so model weights persist across cold starts
#   5. Set RUNPOD_ENDPOINT_URL=https://api.runpod.ai/v2/<endpoint_id>/run in ve-app Vercel env vars
# Original proven base — broad GPU support (sm_35–sm_90), no CUDA kernel compat issues.
# weights_only=False patch removes the need for add_safe_globals (torch 2.4+ only).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch + torchaudio already included in the RunPod base image.
# Upgrade pip only.
RUN pip install --no-cache-dir --upgrade pip

# Copy and install remaining deps (don't reinstall torch — use requirements-docker.txt)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Resemble Enhance: clone and copy package, then patch so deepspeed is optional (inference-only).
# Patches enhancer/train.py, utils/distributed.py, utils/engine.py. Same RUN as clone so cache never yields unpatched tree.
COPY patch_deepspeed_optional.py .
RUN git clone --depth 1 https://github.com/resemble-ai/resemble-enhance.git /tmp/resemble-enhance \
    && cp -r /tmp/resemble-enhance/resemble_enhance /app/ \
    && cp -r /tmp/resemble-enhance/config /app/ \
    && rm -rf /tmp/resemble-enhance \
    && python3 /app/patch_deepspeed_optional.py

# Verify the patched import chain (no deepspeed required). Print full traceback on failure.
ENV PYTHONPATH=/app
COPY check_import.py .
RUN python3 check_import.py

COPY app.py .
COPY handler.py .

# RunPod serverless: no port needed — handler.py is the entrypoint
CMD ["python3", "-u", "handler.py"]
