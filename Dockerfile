# VocalEnhancer worker: Resemble Enhance on RunPod Serverless.
# Build from repo root: docker build -t vocal-enhancer-worker .
#
# Uses RunPod's official PyTorch base image (cuda12.4.1 + PyTorch 2.4) — pre-tested on their GPU fleet.
# PyTorch 2.4+ required for torch.serialization.add_safe_globals (used by Resemble Enhance).
# This avoids "no kernel image available" CUDA errors that occur with the generic nvidia/cuda base.
#
# RunPod Serverless setup:
#   1. Push image to GHCR (GitHub Actions CI handles this)
#   2. Create a Serverless Endpoint in RunPod dashboard, point to this image
#   3. Set env vars: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
#   4. (Optional) Mount a network volume at /runpod-volume and set TORCH_HOME=/runpod-volume/torch_cache
#      so model weights persist across cold starts
#   5. Set RUNPOD_ENDPOINT_URL=https://api.runpod.ai/v2/<endpoint_id>/run in ve-app Vercel env vars
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
