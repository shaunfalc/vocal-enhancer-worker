"""
VocalEnhancer serverless handler for RunPod.
Converts the FastAPI always-on pod to a RunPod serverless worker.

Input payload (from ve-app dispatch.ts):
    { "input": { "job_id": "<uuid>", "input_url": "<signed-url>" } }

The handler updates job status directly in Supabase — ve-app polls Supabase,
so no RunPod polling is required on the app side.

Model caching:
    Set TORCH_HOME env var on the RunPod endpoint to a network volume path
    (e.g. /runpod-volume/torch_cache) so model weights persist across cold starts.
"""
import math
import os
import tempfile
import time
from pathlib import Path

import httpx
import runpod
from supabase import create_client, Client

# ── Config ────────────────────────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# ── Device (resolved once at container startup, stays warm between jobs) ──────
_device = None


def _get_device():
    global _device
    if _device is not None:
        return _device
    import torch
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[handler] Using device: {_device}", flush=True)
    return _device


# ── Pre-warm: load models at container startup to minimise cold-start latency ─
def _prewarm_models():
    """
    Load Resemble Enhance models into memory once at container startup.
    With a RunPod network volume at $TORCH_HOME, weights are read from disk
    (not re-downloaded) on every cold start.
    """
    try:
        import torch
        from resemble_enhance.enhancer.inference import denoise, enhance

        device = _get_device()
        print("[handler] Pre-warming Resemble Enhance models…", flush=True)
        # Trigger model load with a tiny silent waveform
        dummy = torch.zeros(16000).to(device)
        denoise(dummy, 16000, device)
        enhance(dummy, 16000, device, nfe=2, solver="midpoint", lambd=0.1, tau=0.5)
        print("[handler] Models warmed.", flush=True)
    except Exception as e:
        print(f"[handler] Pre-warm failed (non-fatal): {e}", flush=True)


_prewarm_models()

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
PROCESSING_TIMEOUT_SECONDS = 1800       # 30 min (RunPod serverless max is configurable)


# ── Core processing logic (unchanged from app.py) ────────────────────────────
def _process_job(job_id: str, input_url: str) -> dict:
    start_time = time.time()

    # Idempotency: re-fetch job; skip if not in processing state
    r = supabase.table("jobs").select("id, user_id, file_id, status").eq("id", job_id).single().execute()
    job = r.data
    if not job or job.get("status") != "processing":
        return {"skipped": True, "reason": f"job status is '{job.get('status') if job else 'not found'}'"}

    user_id = job["user_id"]
    file_id = job["file_id"]

    fr = supabase.table("files").select("duration_seconds").eq("id", file_id).single().execute()
    duration_seconds = float(fr.data.get("duration_seconds", 0) or 0)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.audio"
        output_path = Path(tmpdir) / "output.wav"

        # Download with size limit
        with httpx.Client(timeout=300.0) as client:
            try:
                head_resp = client.head(input_url)
                content_length = head_resp.headers.get("content-length")
                if content_length and int(content_length) > MAX_DOWNLOAD_BYTES:
                    raise RuntimeError(f"File too large ({int(content_length)} bytes). Maximum is 500 MB.")
            except httpx.HTTPError:
                pass

            downloaded = 0
            with client.stream("GET", input_url) as resp:
                resp.raise_for_status()
                with open(input_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                        downloaded += len(chunk)
                        if downloaded > MAX_DOWNLOAD_BYTES:
                            raise RuntimeError(f"File exceeds 500 MB limit. Aborting.")
                        f.write(chunk)

        import torch
        import torchaudio
        from resemble_enhance.enhancer.inference import denoise, enhance

        dwav, sr = torchaudio.load(str(input_path))
        dwav = dwav.mean(dim=0)
        device = _get_device()
        dwav = dwav.to(device)

        wav_denoised, sr_denoise = denoise(dwav, sr, device)
        wav_denoised = wav_denoised.squeeze(0) if wav_denoised.dim() > 1 else wav_denoised
        wav_denoised = wav_denoised.to(device)
        wav_out, new_sr = enhance(
            wav_denoised, sr_denoise, device,
            nfe=64, solver="midpoint", lambd=0.1, tau=0.5
        )
        wav_np = wav_out.cpu().numpy().squeeze()

        import scipy.io.wavfile as wavfile
        wavfile.write(str(output_path), int(new_sr), wav_np)

        output_path_str = f"{user_id}/{job_id}.wav"
        with open(output_path, "rb") as f:
            supabase.storage.from_("outputs").upload(
                output_path_str,
                f.read(),
                file_options={"content-type": "audio/wav", "upsert": "true"},
            )

    out_file_r = supabase.table("files").insert({
        "user_id": user_id,
        "storage_path": output_path_str,
        "bucket": "outputs",
        "duration_seconds": duration_seconds,
        "mime_type": "audio/wav",
    }).execute()
    output_file_id = out_file_r.data[0]["id"]

    processing_seconds = time.time() - start_time
    supabase.table("jobs").update({
        "status": "completed",
        "output_file_id": output_file_id,
        "processing_seconds": round(processing_seconds, 2),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }).eq("id", job_id).execute()

    return {
        "job_id": job_id,
        "output_file_id": output_file_id,
        "processing_seconds": round(processing_seconds, 2),
    }


# ── RunPod serverless handler ─────────────────────────────────────────────────
def handler(job: dict) -> dict:
    """
    RunPod serverless entry point.
    job["input"] = { "job_id": str, "input_url": str }
    """
    inp = job.get("input", {})
    job_id = inp.get("job_id")
    input_url = inp.get("input_url")

    if not job_id or not input_url:
        return {"error": "job_id and input_url are required"}

    print(f"[handler] Processing job {job_id}", flush=True)

    try:
        result = _process_job(job_id, input_url)
        print(f"[handler] Job {job_id} complete: {result}", flush=True)
        return result
    except Exception as e:
        err_msg = str(e)[:500]
        print(f"[handler] Job {job_id} failed: {err_msg}", flush=True)

        # Update Supabase job status to failed
        try:
            supabase.table("jobs").update({
                "status": "failed",
                "error_message": err_msg,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }).eq("id", job_id).execute()
        except Exception:
            pass

        # Refund credits
        try:
            jr = supabase.table("jobs").select("user_id, file_id").eq("id", job_id).single().execute()
            if jr.data:
                fr = supabase.table("files").select("duration_seconds").eq("id", jr.data["file_id"]).single().execute()
                dur_sec = float((fr.data or {}).get("duration_seconds", 0) or 0)
                dur_min = math.ceil(dur_sec / 60.0)
                supabase.rpc("refund_usage_and_credit", {
                    "p_user_id": jr.data["user_id"],
                    "p_duration_minutes": dur_min,
                }).execute()
        except Exception as refund_err:
            print(f"[handler] Refund failed for job {job_id}: {refund_err}", flush=True)

        # Re-raise so RunPod marks this job as FAILED
        raise


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
