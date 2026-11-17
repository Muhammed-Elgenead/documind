# Deploying DocuMind

This guide puts the app online on three services:

| Part | Host | Why |
|---|---|---|
| Database | [Neon](https://neon.com) (Postgres + pgvector) | Managed Postgres with pgvector on the free plan |
| Backend (FastAPI) | [Railway](https://railway.com) (or [Render](#alternative-render-instead-of-railway)) | Runs the Docker image; usage-based pricing, no hard 512 MB memory cap |
| Frontend (React) | [Vercel](https://vercel.com) | Builds and serves the static site from the repo |

It takes about 30 minutes. You need a GitHub account and a Gemini API key.

> **Memory matters.** The backend runs two local ONNX models (embeddings and re-ranker). With the default batch size of 4 it peaks at roughly **0.9 GB** of RAM while indexing and re-ranking. Measurements are in [decisions.md](decisions.md). That's why the guide uses Railway, where a service can use more than 512 MB and you pay for what it uses (about $10 per GB of RAM per month; the $5/month Hobby plan includes $5 of usage). Check [railway.com/pricing](https://railway.com/pricing) for current rates.

> **LLM quota.** On the Gemini free tier each model allows about 20 requests per day, so a public demo runs out fast. Enable billing on the key's Google Cloud project for real use; each answer costs well under a cent.

---

## 1. Put the code on GitHub

Railway and Vercel deploy from a Git repository. From the project root:

```bash
git init
git add .
git status            # check that .env is NOT listed (it's in .gitignore)
git commit -m "DocuMind"
```

Create an empty repository on GitHub (e.g. `documind`, private is fine), then:

```bash
git remote add origin https://github.com/<you>/documind.git
git branch -M main
git push -u origin main
```

Or with the GitHub CLI: `gh repo create documind --private --source=. --push`.

## 2. Create the database on Neon

1. Sign up at [neon.com](https://neon.com) and create a project. Pick the region closest to where the backend will run, e.g. AWS US East if Railway runs in US East.
2. On the project dashboard click **Connect**.
3. **Turn "Connection pooling" off** and copy the connection string. It looks like this:
   ```
   postgresql://neondb_owner:...@ep-cool-name-123456.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require
   ```
   Use this *direct* string (no `-pooler` in the host name). The pooled one goes through PgBouncer, which doesn't work well with the asyncpg driver's prepared statements. The backend runs a single process with a small connection pool, so it doesn't need the pooler.

Paste the string as is. The backend converts `postgresql://`, `sslmode` and `channel_binding` into what asyncpg expects. You don't need to enable pgvector by hand: the first migration runs `CREATE EXTENSION vector`.

Note: on the free plan Neon suspends the database after 5 minutes without queries. The first request after a pause takes about a second longer.

## 3. Generate a JWT secret

The backend signs login tokens with it. Keep it secret and stable: changing it signs everyone out.

```bash
openssl rand -hex 32
# or: python -c "import secrets; print(secrets.token_hex(32))"
```

## 4. Deploy the backend on Railway

1. Sign up at [railway.com](https://railway.com) with GitHub and choose the Hobby plan.
2. **New Project → Deploy from GitHub repo →** pick your repository.
3. Open the new service, then **Settings**:
   - **Source → Root Directory:** `backend`. Railway then finds `backend/Dockerfile` and builds it.
   - **Deploy → Healthcheck Path:** `/health`
   - **Networking → Generate Domain** with target port `8000`. You get a URL like `https://documind-production.up.railway.app`.
4. **Variables** (use the Raw Editor and paste, then fill in):
   ```
   PORT=8000
   DATABASE_URL=<the Neon connection string from step 2>
   JWT_SECRET=<the secret from step 3>
   GEMINI_API_KEY=<your key>
   LLM_MODEL=gemini-3.8-flash
   CORS_ORIGINS=["http://localhost:5173"]
   RETRIEVAL_MODE=hybrid_rerank
   EMBEDDING_BATCH_SIZE=4
   RERANK_BATCH_SIZE=4
   CHAT_RATE_LIMIT=10/minute;200/day
   ```
   `CORS_ORIGINS` gets the real Vercel address in step 6.
5. Railway deploys automatically. The first build takes several minutes because it downloads the two models into the image. In **Deployments → View logs** you should see:
   ```
   Running upgrade  -> 0001 ... Running upgrade 0003 -> 0004
   Uvicorn running on http://0.0.0.0:8000
   ```
6. Check it:
   ```bash
   curl https://<your-api-domain>/health        # {"status":"healthy","database":"ok"}
   curl -i https://<your-api-domain>/documents  # 401: sign-in required
   ```

## 5. Deploy the frontend on Vercel

1. Sign up at [vercel.com](https://vercel.com) with GitHub.
2. **Add New → Project →** import the repository.
3. **Root Directory:** `frontend`. The framework (Vite), build command and output folder come from `frontend/vercel.json`.
4. **Environment Variables:** add
   ```
   VITE_API_URL = https://<your-api-domain>
   ```
   Use your Railway URL, with no trailing slash. Vite bakes it into the build, so changing it later needs a redeploy.
5. **Deploy.** You get a URL like `https://documind-abc.vercel.app`.

## 6. Allow the frontend to call the API (CORS)

Back in Railway → **Variables**, set the exact Vercel origin (scheme + host, no path, no trailing slash):

```
CORS_ORIGINS=["https://documind-abc.vercel.app"]
```

Optional: to also allow Vercel preview deployments (one URL per branch or PR), add a regex that matches your project's preview URLs:

```
CORS_ORIGIN_REGEX=https://documind-[a-z0-9-]+\.vercel\.app
```

Railway redeploys when variables change.

## 7. Try it

Open the Vercel URL:

1. **Create an account**: you're redirected to sign-in, so pick "Create an account".
2. **Documents**: drop a PDF. It shows `processing`, then `ready`.
3. **New chat**: ask something the PDF answers, then click a citation chip.

That's your live link. Pushing to `main` redeploys both the backend (Railway) and the frontend (Vercel).

---

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Browser console: *blocked by CORS policy* | `CORS_ORIGINS` doesn't exactly match the page's origin. Check `https://`, no trailing slash, and the JSON list format `["..."]`. |
| Every request returns 401 after a redeploy | `JWT_SECRET` changed or isn't set (a random one is used per start). Set it once and keep it. |
| Backend crashes at start with a database error | Wrong `DATABASE_URL`, or the pooled (`-pooler`) host. Use Neon's direct string. |
| Service restarts while indexing or answering (out of memory) | Lower `EMBEDDING_BATCH_SIZE`/`RERANK_BATCH_SIZE` to 1, or set `RETRIEVAL_MODE=hybrid` (no re-ranker, about 300 MB less). |
| Answers fail with "rate limited" | Gemini free-tier quota. Enable billing, or switch `LLM_MODEL` to another Flash model with its own quota. |
| Chat says "Too many requests. Please try again in N s." | DocuMind's own per-user limit (`CHAT_RATE_LIMIT`). Raise it if needed. |
| A document stays `failed` with "Processing was interrupted" | The backend restarted while indexing (deploy or crash). Upload it again. |
| Frontend loads but shows "backend unavailable" | `VITE_API_URL` is wrong or missing. Fix it in Vercel and redeploy. |

## Things to know about this setup

- **Single backend instance.** Rate limits and background indexing live inside the process, and migrations run at start. Don't scale the service to several replicas without moving those out first (see "What I'd do next" in the README).
- **Uploaded files are temporary.** The container's disk is wiped on every deploy. That's fine: a file is only read while it's being indexed, and its text and embeddings then live in Postgres.
- **Secrets** live only in the Railway and Vercel dashboards and your local `.env`, never in the repo.

## Free alternative: Hugging Face Spaces instead of Railway

Free CPU hardware has enough memory for both models. The trade-offs: the Space sleeps after a period without visitors (the next request wakes it, which takes a minute or two), and it must be **public** so the browser can call it.

1. On [huggingface.co](https://huggingface.co): **New → Space**. Pick a name (e.g. `documind-api`), **SDK: Docker** (blank template), hardware **CPU basic (free)**, visibility **Public**.
2. In the Space: **Settings → Variables and secrets**. Add the variables from step 4 above as **secrets** (`DATABASE_URL`, `JWT_SECRET`, `GEMINI_API_KEY`) and **variables** (the rest). Skip `PORT`; `backend/README.md` already sets `app_port: 8000`.
3. Create an access token: **Settings → Access Tokens → Create new token**, type **Write**.
4. In the GitHub repo: **Settings → Secrets and variables → Actions**. Add the secret `HF_TOKEN` (the token) and, on the **Variables** tab, `HF_SPACE` = `<hf-username>/<space-name>`.
5. In GitHub **Actions**, run **Deploy API to Hugging Face** (**Run workflow**). After that it runs by itself whenever `backend/` changes on `main`. It pushes `backend/` to the Space, and Hugging Face builds the Docker image.
6. The API is at `https://<hf-username>-<space-name>.hf.space`. Use that as `VITE_API_URL` on Vercel (step 5), and put the Vercel URL in the Space's `CORS_ORIGINS` (step 6).

## Alternative: Render instead of Railway

Steps 1–3 and 5–7 stay the same. Instead of step 4:

1. On [render.com](https://render.com): **New → Web Service →** connect the repository.
2. **Root Directory** `backend`, **Language/Runtime** Docker. Render detects the Dockerfile and sets `$PORT` itself.
3. **Instance type:** pick one with **2 GB RAM** (about $25/month; Render is renaming its plans, so check [render.com/pricing](https://render.com/pricing)). The 512 MB tiers are too small for the default setup. To fit 512 MB, set `RETRIEVAL_MODE=hybrid`, `EMBEDDING_BATCH_SIZE=1` and `RERANK_BATCH_SIZE=1`; peak memory is then about 420 MB, which works but is tight.
4. **Health Check Path** `/health`, and the same environment variables as in step 4 above.
5. The free instance type sleeps after 15 minutes without traffic, and the first request then waits for a cold start of about a minute. Paid instance types don't sleep.

## Self-hosting on one machine instead

`docker-compose.prod.yml` runs Postgres, the API and an nginx-served frontend together:

```bash
# .env needs POSTGRES_PASSWORD, JWT_SECRET and GEMINI_API_KEY
docker compose -f docker-compose.prod.yml up --build -d
# -> http://localhost:8080
```
