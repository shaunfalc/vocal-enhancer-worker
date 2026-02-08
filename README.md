# VocalEnhancer Worker

Python worker for [VocalEnhancer](https://github.com/your-org/vocal-enhancer): accepts enhancement jobs, runs [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) (denoise + enhance), and uploads results to Supabase. Designed to run on an **always-on RunPod dedicated pod** with GPU.

## Contract

- **Endpoint**: `POST /run` (and `GET /health` for monitoring).
- **Request body**: `{ "input": { "job_id": "<uuid>", "input_url": "<signed-url>" } }`.
- **Auth**: `Authorization: Bearer <WORKER_SECRET>` (or `RUNPOD_API_KEY` — same value as in the app).
- **Response**: **202 Accepted** with `{ "status": "accepted", "job_id": "..." }`. Processing runs in the background so the request returns within RunPod’s 100-second proxy timeout.

The app dispatcher (cron or immediate) sends jobs to this worker; users poll `GET /api/jobs/:id` until the job is completed or failed.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SUPABASE_URL` | Yes | Supabase project URL. |
| `SUPABASE_SERVICE_ROLE_KEY` | Yes | Service role key (bypasses RLS). |
| `WORKER_SECRET` or `RUNPOD_API_KEY` | Yes | Shared secret for `Authorization: Bearer`; set the same value in the app as `RUNPOD_API_KEY`. |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
export SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... RUNPOD_API_KEY=...
uvicorn app:app --host 0.0.0.0 --port 8000
```

- Health: `GET http://localhost:8000/health`
- Run: `POST http://localhost:8000/run` with JSON body and `Authorization: Bearer <RUNPOD_API_KEY>`.

## Docker (for RunPod)

Build from repo root (no wrapper directory):

```bash
docker build -t vocal-enhancer-worker .
docker run --env-file .env -p 8000:8000 vocal-enhancer-worker
```

## RunPod always-on pod setup

1. **Build and push** the image from this repo root:
   ```bash
   docker build -t your-registry/vocal-enhancer-worker:latest .
   docker push your-registry/vocal-enhancer-worker:latest
   ```
2. **Create a pod** in [RunPod Console](https://console.runpod.io/pod/create):
   - **GPU**: e.g. RTX 4090, A40, or T4 (Resemble Enhance uses CUDA).
   - **Container image**: `your-registry/vocal-enhancer-worker:latest`.
   - **Expose HTTP Ports**: `8000` (so the worker is reachable at `https://<POD_ID>-8000.proxy.runpod.net`).
   - **Environment variables**: Add `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, and `RUNPOD_API_KEY` (or `WORKER_SECRET`) with your values.
   - Start command can stay default (image CMD runs uvicorn).
3. **Note the Pod ID** from the RunPod console (e.g. `abc123xyz`) once the pod is running.
4. **Configure the app (ve-app)**:
   - Set `RUNPOD_ENDPOINT_URL=https://<POD_ID>-8000.proxy.runpod.net/run` (replace `<POD_ID>`).
   - Set `RUNPOD_API_KEY` to the same secret you set on the worker.

After that, the app’s cron dispatch will send jobs to this worker; users poll `GET /api/jobs/:id` until completion.

## Supported input formats

WAV and MP3 (via torchaudio). M4A can be added later (e.g. via ffmpeg in the image).

## End-to-end testing

1. **Start the worker** (local or pod): `uvicorn app:app --host 0.0.0.0 --port 8000` with env vars set.
2. **Start the app (ve-app)** with `RUNPOD_ENDPOINT_URL` pointing at the worker (e.g. `http://localhost:8000/run` for local) and `RUNPOD_API_KEY` set.
3. **In the app**: Sign in, upload an audio file (WAV or MP3), click Enhance.
4. **Trigger dispatch**: Call `GET /api/cron/dispatch?secret=<CRON_SECRET>` or wait for the cron (every minute).
5. **Poll** `GET /api/jobs/:id` until `status` is `completed` or `failed`.
6. **Verify**: When completed, the response includes `preview_url` and `download_url`; open or download to confirm the enhanced audio.

## License

MIT.
