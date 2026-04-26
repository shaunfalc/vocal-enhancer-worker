"""
VocalEnhancer serverless handler for RunPod.

Supports two modes:
  1. Marketplace mode (callback_url present in input):
     - Reads input_url directly from payload
     - Uploads output to Supabase storage
     - Calls callback_url to update ai_jobs table
  2. Legacy ve-app mode (no callback_url):
     - Reads job/file info from Supabase "jobs" + "files" tables
     - Updates those tables directly

Input payload:
    { "input": { "job_id": "<uuid>", "input_url": "<signed-url>", "callback_url?": "..." } }

Model caching:
    Set TORCH_HOME env var on the RunPod endpoint to a network volume path
    (e.g. /runpod-volume/torch_cache) so model weights persist across cold starts.
"""
import hashlib
import hmac
import json
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

# HMAC secret for signing marketplace callbacks. Must match VE_WEBHOOK_SECRET
# on the marketplace (vocalpresets.com) side. Falls back to legacy name.
MARKETPLACE_WEBHOOK_SECRET = (
    os.environ.get("MARKETPLACE_WEBHOOK_SECRET")
    or os.environ.get("VE_WEBHOOK_SECRET")
    or ""
)

# Output bucket for marketplace mode. The marketplace Supabase has private
# `uploads` (input) + `outputs` (output) buckets — see
# 20260324180000_rename_storage_buckets_to_match_ve.sql.
MARKETPLACE_OUTPUT_BUCKET = os.environ.get("MARKETPLACE_OUTPUT_BUCKET", "outputs")
# 7 days — matches the "files available for 7 days" UX copy on result screen.
MARKETPLACE_SIGNED_URL_TTL_SECONDS = 60 * 60 * 24 * 7

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


# ── Enhancement pipeline (shared by both modes) ──────────────────────────────
def _enhance_audio(input_path: Path, output_path: Path, do_denoise: bool = True, do_derev: bool = True):
    """
    Run the requested pipeline on an audio file. Writes WAV to output_path.

    do_denoise / do_derev semantics (mirroring the marketplace UI toggles):
      - denoise=True,  derev=True  → full pipeline (denoise → enhance)
      - denoise=True,  derev=False → denoise only (no dereverb / super-res)
      - denoise=False, derev=True  → enhance only (resemble's enhance also denoises internally)
      - denoise=False, derev=False → caller should reject; treated as a no-op passthrough copy
    """
    import torch
    import torchaudio
    import numpy as np
    import scipy.io.wavfile as wavfile
    from resemble_enhance.enhancer.inference import denoise, enhance

    dwav, sr = torchaudio.load(str(input_path))
    dwav = dwav.mean(dim=0)
    device = _get_device()
    dwav = dwav.to(device)

    if not do_denoise and not do_derev:
        # Defensive — UI gates against this, but if it slips through, just copy.
        print("[enhance] No-op (both toggles off); writing input through.", flush=True)
        wav_np = np.atleast_1d(dwav.cpu().numpy())
        wavfile.write(str(output_path), int(sr), wav_np)
        return

    if do_derev:
        # Full pipeline: optional pre-denoise + enhance (which itself denoises + dereverbs)
        if do_denoise:
            print("[enhance] Denoising (pre-pass)…", flush=True)
            wav_pre, sr_pre = denoise(dwav, sr, device)
            wav_pre = wav_pre.squeeze(0) if wav_pre.dim() > 1 else wav_pre
            wav_pre = wav_pre.to(device)
        else:
            wav_pre, sr_pre = dwav, sr

        print("[enhance] Enhancing (denoise + dereverb)…", flush=True)
        wav_out, new_sr = enhance(
            wav_pre, sr_pre, device,
            nfe=64, solver="midpoint", lambd=0.1, tau=0.5,
        )
    else:
        # Denoise only
        print("[enhance] Denoising only…", flush=True)
        wav_out, new_sr = denoise(dwav, sr, device)
        wav_out = wav_out.squeeze(0) if wav_out.dim() > 1 else wav_out

    print("[enhance] Saving…", flush=True)
    wav_out = wav_out.cpu()
    wav_np = np.atleast_1d(wav_out.numpy())
    sr_out = int(new_sr.item()) if hasattr(new_sr, 'item') else int(new_sr)
    wavfile.write(str(output_path), sr_out, wav_np)


# ── Marketplace callback (signed) ─────────────────────────────────────────────
def _post_marketplace_callback(callback_url: str, payload: dict) -> None:
    """POST a JSON payload to the marketplace callback, signed with HMAC-SHA256."""
    body = json.dumps(payload, separators=(",", ":"))
    headers = {"Content-Type": "application/json"}
    if MARKETPLACE_WEBHOOK_SECRET:
        sig = hmac.new(
            MARKETPLACE_WEBHOOK_SECRET.encode("utf-8"),
            body.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        headers["X-Webhook-Signature"] = sig
    else:
        print(
            "[marketplace] WARNING: MARKETPLACE_WEBHOOK_SECRET unset — "
            "callback will be rejected as 401 by marketplace.",
            flush=True,
        )
    with httpx.Client(timeout=15.0) as client:
        resp = client.post(callback_url, content=body, headers=headers)
        print(f"[marketplace] Callback {callback_url} → {resp.status_code}", flush=True)


def _download_audio(input_url: str, input_path: Path):
    """Download audio file with size limit."""
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
                        raise RuntimeError("File exceeds 500 MB limit. Aborting.")
                    f.write(chunk)


# ── Marketplace mode (ai_jobs table, callback URL) ───────────────────────────
def _process_marketplace_job(
    job_id: str,
    input_url: str,
    callback_url: str,
    do_denoise: bool = True,
    do_derev: bool = True,
) -> dict:
    """Process a job from the marketplace. Uses callback_url to report results."""
    start_time = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.audio"
        output_path = Path(tmpdir) / "output.wav"

        _download_audio(input_url, input_path)
        _enhance_audio(input_path, output_path, do_denoise=do_denoise, do_derev=do_derev)

        # Upload to private `outputs` bucket on the marketplace Supabase.
        # Path matches the legacy ve-app convention so RLS policies line up.
        output_storage_path = f"{job_id}/output.wav"
        with open(output_path, "rb") as f:
            supabase.storage.from_(MARKETPLACE_OUTPUT_BUCKET).upload(
                output_storage_path,
                f.read(),
                file_options={"content-type": "audio/wav", "upsert": "true"},
            )

    # `outputs` is private — issue a signed URL that survives the 7-day result window.
    signed = supabase.storage.from_(MARKETPLACE_OUTPUT_BUCKET).create_signed_url(
        output_storage_path, MARKETPLACE_SIGNED_URL_TTL_SECONDS,
    )
    output_url = (
        signed.get("signedURL")
        or signed.get("signed_url")
        or signed.get("signedUrl")
        if isinstance(signed, dict)
        else None
    )
    if not output_url:
        raise RuntimeError(f"Failed to create signed URL for output: {signed!r}")

    processing_seconds = round(time.time() - start_time, 2)
    print(f"[marketplace] Job {job_id} enhanced in {processing_seconds}s", flush=True)

    # Signed callback to marketplace to update ai_jobs
    try:
        _post_marketplace_callback(callback_url, {
            "job_id": job_id,
            "status": "completed",
            "output_url": output_url,
        })
    except Exception as e:
        print(f"[marketplace] Callback failed (non-fatal): {e}", flush=True)
        # Non-fatal: the marketplace also polls RunPod status as fallback

    return {
        "job_id": job_id,
        "output_url": output_url,
        "processing_seconds": processing_seconds,
    }


# ── Legacy ve-app mode (jobs + files tables) ─────────────────────────────────
def _process_legacy_job(job_id: str, input_url: str) -> dict:
    """Process a job from ve-app. Updates jobs/files tables directly in Supabase."""
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

    def _set_stage(stage: str):
        """Write processing stage to Supabase so the frontend can show real-time progress."""
        try:
            supabase.table("jobs").update({"stage": stage}).eq("id", job_id).execute()
            print(f"[stage] {stage}", flush=True)
        except Exception as e:
            print(f"[stage] failed to set stage={stage}: {e}", flush=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.audio"
        output_path = Path(tmpdir) / "output.wav"

        _set_stage("downloading")
        _download_audio(input_url, input_path)

        _set_stage("denoising")
        _enhance_audio(input_path, output_path)

        _set_stage("saving")
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
    job["input"] = { "job_id": str, "input_url": str, "callback_url?": str }

    If callback_url is present → marketplace mode (ai_jobs table, callback).
    Otherwise → legacy ve-app mode (jobs + files tables).
    """
    inp = job.get("input", {})
    job_id = inp.get("job_id")
    input_url = inp.get("input_url")
    callback_url = inp.get("callback_url")
    # Marketplace UI toggles. Default true to preserve legacy "always full pipeline" behavior.
    do_denoise = bool(inp.get("denoise", True))
    do_derev = bool(inp.get("derev", True))

    if not job_id or not input_url:
        return {"error": "job_id and input_url are required"}

    is_marketplace = bool(callback_url)
    mode = "marketplace" if is_marketplace else "ve-app"
    print(
        f"[handler] Processing job {job_id} (mode={mode}, denoise={do_denoise}, derev={do_derev})",
        flush=True,
    )

    try:
        if is_marketplace:
            result = _process_marketplace_job(
                job_id, input_url, callback_url,
                do_denoise=do_denoise, do_derev=do_derev,
            )
        else:
            result = _process_legacy_job(job_id, input_url)

        print(f"[handler] Job {job_id} complete: {result}", flush=True)
        return result

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        err_msg = f"{str(e)} ||| TRACE: {tb}"[:3000]
        print(f"[handler] Job {job_id} failed: {str(e)}", flush=True)
        print(f"[handler] Traceback:\n{tb}", flush=True)

        if is_marketplace:
            # Signed callback with failure so marketplace can refund + show error
            try:
                _post_marketplace_callback(callback_url, {
                    "job_id": job_id,
                    "status": "failed",
                    "error": str(e)[:500],
                })
            except Exception:
                pass
        else:
            # Legacy: update jobs table directly
            try:
                supabase.table("jobs").update({
                    "status": "failed",
                    "error_message": err_msg,
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }).eq("id", job_id).execute()
            except Exception:
                pass

            # Legacy: refund credits
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
