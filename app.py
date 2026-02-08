"""
VocalEnhancer worker: accepts enhancement jobs, runs Resemble Enhance, uploads to Supabase.
Returns 202 Accepted immediately and processes in background (RunPod proxy 100s timeout).
"""
import math
import os
import tempfile
import threading
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase import create_client, Client

# Resemble Enhance (lazy import after torch to set device)
_device = None

def _get_device():
    global _device
    if _device is not None:
        return _device
    import torch
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


# ----- Config -----
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
# Prefer WORKER_SECRET so RunPod-injected RUNPOD_API_KEY doesn't override your app key
WORKER_SECRET = os.environ.get("WORKER_SECRET") or os.environ.get("RUNPOD_API_KEY")
_WORKER_SECRET_SOURCE = "WORKER_SECRET" if os.environ.get("WORKER_SECRET") else "RUNPOD_API_KEY"

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
if not WORKER_SECRET:
    raise RuntimeError("WORKER_SECRET or RUNPOD_API_KEY must be set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
security = HTTPBearer(auto_error=False)


def verify_bearer(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> None:
    import logging
    logger = logging.getLogger("uvicorn")
    
    if not WORKER_SECRET:
        logger.error("WORKER_SECRET or RUNPOD_API_KEY is not set in environment variables!")
        raise HTTPException(status_code=500, detail="Worker configuration error")

    if not credentials:
        logger.warning("No credentials provided in Authorization header")
        raise HTTPException(status_code=401, detail="Unauthorized - No credentials")

    if credentials.credentials != WORKER_SECRET:
        logger.warning(f"Token mismatch. Expected length: {len(WORKER_SECRET)}, Received length: {len(credentials.credentials)}")
        # Optional: print first few chars for debugging (remove in production)
        # logger.warning(f"Expected start: {WORKER_SECRET[:4]}..., Received start: {credentials.credentials[:4]}...")
        raise HTTPException(status_code=401, detail="Unauthorized - Invalid token")


# ----- FastAPI app -----
app = FastAPI(title="VocalEnhancer Worker", version="0.1.0")


@app.on_event("startup")
def _log_auth_config() -> None:
    """Log auth config at startup (no secret value) so you can verify from pod logs."""
    import sys
    msg = (
        f"Worker auth: {_WORKER_SECRET_SOURCE} set, length={len(WORKER_SECRET)}. "
        "App must send Authorization: Bearer <same value>. "
        "If you get 401, set WORKER_SECRET on this pod to match ve-app RUNPOD_API_KEY."
    )
    print(msg, flush=True)
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


@app.get("/")
def root():
    """Root endpoint for load balancer/proxy probes; use GET /health for monitoring."""
    return {"service": "VocalEnhancer Worker", "health": "/health", "run": "POST /run"}


@app.get("/health")
def health():
    """Uptime/monitoring: returns 200 when worker is up."""
    return {"status": "ok"}


@app.post("/run", dependencies=[Depends(verify_bearer)])
async def run(request: Request):
    """
    Accept enhancement job. Body: { "input": { "job_id": "<uuid>", "input_url": "<signed-url>" } }.
    Returns 202 Accepted immediately; processing runs in background.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    inp = body.get("input") or body
    job_id = inp.get("job_id")
    input_url = inp.get("input_url")
    if not job_id or not input_url:
        raise HTTPException(status_code=400, detail="job_id and input_url required")

    # Spawn background task and return immediately (stay under 100s proxy timeout)
    thread = threading.Thread(target=_process_job, args=(job_id, input_url), daemon=True)
    thread.start()

    from fastapi.responses import JSONResponse
    return JSONResponse(
        content={"status": "accepted", "job_id": job_id},
        status_code=202,
    )


def _process_job(job_id: str, input_url: str) -> None:
    start_time = time.time()
    try:
        # Idempotency: re-fetch job; if not processing, skip
        r = supabase.table("jobs").select("id, user_id, file_id, status").eq("id", job_id).single().execute()
        job = r.data
        if not job or job.get("status") != "processing":
            return

        user_id = job["user_id"]
        file_id = job["file_id"]

        # Input file duration for usage
        fr = supabase.table("files").select("duration_seconds").eq("id", file_id).single().execute()
        duration_seconds = float(fr.data.get("duration_seconds", 0) or 0)
        duration_minutes = duration_seconds / 60.0

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.audio"
            output_path = Path(tmpdir) / "output.wav"

            # Download
            with httpx.Client(timeout=300.0) as client:
                resp = client.get(input_url)
                resp.raise_for_status()
                input_path.write_bytes(resp.content)

            # Load with torchaudio (WAV/MP3)
            import torch
            import torchaudio
            from resemble_enhance.enhancer.inference import denoise, enhance

            dwav, sr = torchaudio.load(str(input_path))
            dwav = dwav.mean(dim=0).unsqueeze(0)  # mono
            device = _get_device()
            dwav = dwav.to(device)

            # Denoise then enhance (same as Resemble app.py: lambd=0.9 for denoising)
            wav_denoised, sr_denoise = denoise(dwav, sr, device)
            wav_out, new_sr = enhance(
                wav_denoised, sr_denoise, device,
                nfe=64, solver="midpoint", lambd=0.1, tau=0.5
            )
            wav_np = wav_out.cpu().numpy().squeeze()

            # Write WAV
            import scipy.io.wavfile as wavfile
            wavfile.write(str(output_path), int(new_sr), wav_np)

            # Upload to Supabase outputs
            output_path_str = f"{user_id}/{job_id}.wav"
            with open(output_path, "rb") as f:
                supabase.storage.from_("outputs").upload(
                    output_path_str,
                    f.read(),
                    file_options={"content-type": "audio/wav", "upsert": "true"}
                )

        # Insert output file row
        out_file_r = supabase.table("files").insert({
            "user_id": user_id,
            "storage_path": output_path_str,
            "bucket": "outputs",
            "duration_seconds": duration_seconds,
            "mime_type": "audio/wav",
        }).select("id").execute()
        output_file_id = out_file_r.data[0]["id"]

        processing_seconds = time.time() - start_time
        supabase.table("jobs").update({
            "status": "completed",
            "output_file_id": output_file_id,
            "processing_seconds": round(processing_seconds, 2),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }).eq("id", job_id).execute()

    except Exception as e:
        import logging
        err_msg = str(e)[:500]
        try:
            supabase.table("jobs").update({
                "status": "failed",
                "error_message": err_msg,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }).eq("id", job_id).execute()
        except Exception:
            pass
        # Refund credits (already deducted at dispatch)
        try:
            jr = supabase.table("jobs").select("user_id, file_id").eq("id", job_id).single().execute()
            if jr.data:
                fr = supabase.table("files").select("duration_seconds").eq("id", jr.data["file_id"]).single().execute()
                dur_sec = float((fr.data or {}).get("duration_seconds", 0) or 0)
                dur_min = math.ceil(dur_sec / 60.0)
                supabase.rpc("refund_usage_and_credit", {"p_user_id": jr.data["user_id"], "p_duration_minutes": dur_min}).execute()
        except Exception as rpc_err:
            logging.warning("refund_usage_and_credit failed, retrying once: job_id=%s err=%s", job_id, rpc_err)
            try:
                jr = supabase.table("jobs").select("user_id, file_id").eq("id", job_id).single().execute()
                if jr.data:
                    fr = supabase.table("files").select("duration_seconds").eq("id", jr.data["file_id"]).single().execute()
                    dur_sec = float((fr.data or {}).get("duration_seconds", 0) or 0)
                    dur_min = math.ceil(dur_sec / 60.0)
                    supabase.rpc("refund_usage_and_credit", {"p_user_id": jr.data["user_id"], "p_duration_minutes": dur_min}).execute()
            except Exception:
                pass
        raise
