# VocalEnhancer Worker

Python worker for VocalEnhancer: accepts enhancement jobs, runs [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance) (denoise + enhance), and uploads results to Supabase. Designed to run on an **always-on RunPod dedicated pod** with GPU.

## Push to GitHub

From the repo root:

```bash
git remote add origin https://github.com/YOUR_ORG/vocal-enhancer-worker.git
git branch -M main
git push -u origin main
```

Or with SSH: `git@github.com:YOUR_ORG/vocal-enhancer-worker.git`. Create the repository on GitHub first (empty, no README).

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
| `WORKER_SECRET` (recommended) or `RUNPOD_API_KEY` | Yes | Shared secret for `Authorization: Bearer`. On RunPod, set **WORKER_SECRET** to the same value as `RUNPOD_API_KEY` in ve-app so RunPod-injected vars don't override it. |

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

## CI/CD (GitHub Actions)

A workflow builds and pushes the Docker image to **GitHub Container Registry** on every push to `main` and on releases. Image: `ghcr.io/shaunfalc/vocal-enhancer-worker:latest`.

- **After the first successful run**: In GitHub go to the repo → **Packages** (right sidebar), open **vocal-enhancer-worker**, then **Package settings** → set **Visibility** to **Public** so RunPod can pull without credentials.
- For RunPod, use container image: `ghcr.io/shaunfalc/vocal-enhancer-worker:latest`.

## Docker (for RunPod)

**If this worker is its own Git repo** (root = this folder):

```bash
docker build -t vocal-enhancer-worker .
docker run --env-file .env -p 8000:8000 vocal-enhancer-worker
```

**If this worker lives inside a monorepo** (e.g. `Vocal Enhancer/vocal-enhancer-worker`), from the **monorepo root**:

```bash
docker build -t vocal-enhancer-worker -f vocal-enhancer-worker/Dockerfile vocal-enhancer-worker
docker run --env-file vocal-enhancer-worker/.env -p 8000:8000 vocal-enhancer-worker
```

## Redeploy after code changes (e.g. new dependency like tqdm)

**Option A — Use CI/CD (easiest)**  
1. Commit and push your changes to the branch that triggers the workflow (e.g. `main`).
2. Wait for the GitHub Actions workflow to build and push the image to `ghcr.io/shaunfalc/vocal-enhancer-worker:latest`.
3. In **RunPod Console** → your pod → **Restart** (or stop and start). The pod will pull the new image on start.

**Option B — Build and push manually**  
1. Build the image (from monorepo root, or from this directory if this is the repo root — see Docker section above).
2. Tag for your registry, e.g. Docker Hub:  
   `docker tag vocal-enhancer-worker YOUR_DOCKERHUB_USER/vocal-enhancer-worker:latest`
3. Push:  
   `docker push YOUR_DOCKERHUB_USER/vocal-enhancer-worker:latest`
4. In RunPod, if your pod uses a custom image URL, restart the pod so it pulls the new tag. If RunPod is set to `ghcr.io/shaunfalc/vocal-enhancer-worker:latest`, use Option A instead.

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
   - **Environment variables**: Add `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, and **`WORKER_SECRET`** (recommended). Set `WORKER_SECRET` to the **exact same value** as `RUNPOD_API_KEY` in your ve-app `.env.local`. Do not rely on `RUNPOD_API_KEY` on the pod—RunPod may inject a different value, which causes 401 Unauthorized.
   - Start command can stay default (image CMD runs uvicorn).
3. **Note the Pod ID** from the RunPod console (e.g. `abc123xyz`) once the pod is running.
4. **Configure the app (ve-app)**:
   - Set `RUNPOD_ENDPOINT_URL=https://<POD_ID>-8000.proxy.runpod.net/run` (replace `<POD_ID>`).
   - Set `RUNPOD_API_KEY` to the same secret you set as `WORKER_SECRET` on the worker.

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
