# RAG Knowledge Assistant

A retrieval-augmented AI backend built with FastAPI and OpenRouter. Upload
any documents — notes, manuals, papers, reports — and have a grounded,
multi-turn conversation about their content, with cited source passages
returned alongside every answer.

What started as a clean service-oriented API layer over OpenRouter has
grown into a full RAG pipeline: document ingestion, chunking, embedding,
re-ranked retrieval, and context-aware chat, all behind a small, typed
FastAPI surface.

## Current Features

- Document ingestion pipeline: PDF upload → text extraction → chunking →
  embedding → vector indexing, run as a background job
- Job tracking (SQLite-backed) so you can poll ingestion status and it
  survives a restart
- Retrieval-augmented `/chat` endpoint — answers are grounded in the most
  relevant retrieved chunks, with those source chunks returned alongside
  the answer
- **LLM-based re-ranking** — over-fetches candidates from the vector
  store, then has the LLM re-score them for actual relevance before
  keeping the top k, catching cases where a chunk is semantically close
  to the query but doesn't really answer it
- **Multi-turn chat history** — conversations persist per `session_id`,
  and follow-up questions ("what about last year?") are rewritten into
  standalone queries before retrieval, so vector search isn't thrown off
  by missing context
- A minimal built-in frontend (`/`) — upload documents and chat with them
  in the browser, no separate hosting or build step required
- OpenRouter integration for multi-model access (swap models via one env
  var — GPT, Claude, Llama, etc.)
- Retry with exponential backoff on embedding/completion calls
- Clean service-oriented architecture (`api/`, `rag/`, `services/`,
  `schemas/`)
- Environment-based configuration (`.env` support)
- CORS configuration and a `/health` endpoint for service monitoring
- Test suite (pytest, 23 tests) with mocked LLM calls, and ruff for linting
- Dockerfile + docker-compose for one-command deployment

## Architecture

```
Upload (PDF)
    │
    ▼
POST /upload ──► save to disk ──► create job (SQLite)
                                                │
                                                ▼    
store in Chroma ◄── embed  ◄── extract text ◄── queue background task  
    │
    ▼
(poll for job status) ───────────► GET /status/{job_id}
                                            │
                                            ▼
                            Question (+ optional session_id)
                                            │
                                            ▼
                                load chat history for session
                                            │
                                            ▼
                         condense question using history (if follow-up)
                                            │
                                            ▼
                                 embed condensed question
                                            │
                                            ▼
                    vector search (Chroma) ──► over-fetch candidates
                                            │
                                            ▼
                          LLM re-ranks candidates ──► top-k chunks
                                            │
                                            ▼
                    LLM answers original question using chunks + history
                                            │
                                            ▼
          save turn to history ──► { answer, sources, session_id }
```

**Design choices worth calling out:**

- `rag/index.py` holds a single shared Chroma `collection` object, imported
  by both the ingestion pipeline and retrieval, so the two paths can never
  end up pointed at different collections.
- Job state and chat history both live in the same SQLite database, not
  memory, so they survive a restart.
- Re-ranking and question-condensing both degrade gracefully — if the LLM
  call fails or returns something unusable, the pipeline falls back to
  plain vector-search order / the original question rather than erroring
  out the whole request.
- Chunking is token-based via `tiktoken` when available, with a
  character-based fallback if the tokenizer can't be loaded (e.g. offline
  environments).
- Uploaded filenames never touch filesystem paths directly (only a
  server-generated UUID + validated extension), ruling out path traversal
  via a malicious filename.

## API

| Endpoint           | Method | Description                                                                         |
| ------------------ | ------ | ----------------------------------------------------------------------------------- |
| `/health`          | GET    | Liveness check                                                                      |
| `/upload`          | POST   | Upload a document (`multipart/form-data`, field `file`). Returns `{job_id, status}` |
| `/status/{job_id}` | GET    | Poll ingestion status: `pending` → `processing` → `done`/`failed`                   |
| `/chat`            | POST   | Ask a question grounded in ingested documents, optionally continuing a session      |

### Example Request

```
POST /chat
```

```json
{
  "question": "What does the document say about onboarding new employees?",
  "k": 4
}
```

```json
{
  "answer": "According to the handbook, new employees complete a two-week...",
  "sources": [
    "...onboarding period covers benefits enrollment, system access...",
    "...managers are expected to assign a buddy for the first 30 days..."
  ],
  "session_id": "3f2a9e1e-8b7a-4b1a-9c2e-1a2b3c4d5e6f"
}
```

Pass `session_id` back on the next request to continue the conversation —
follow-ups like `"what about for remote hires?"` will be resolved against
the chat history automatically:

```json
{
  "question": "what about for remote hires?",
  "session_id": "3f2a9e1e-8b7a-4b1a-9c2e-1a2b3c4d5e6f"
}
```

Interactive docs available at `/docs` once the server is running.

## Tech Stack

- FastAPI
- OpenRouter (OpenAI-compatible API) via the OpenAI Python SDK
- ChromaDB for vector storage
- Pydantic
- SQLite for job tracking and chat history
- Python 3.12+
- Frontend: single-file HTML/CSS/vanilla JS, served by FastAPI's
  `StaticFiles` — no build step, no separate deploy

## Running Locally

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload
```

Open `http://localhost:8000` for the built-in UI, or `http://localhost:8000/docs` to use the API directly.

## Running with Docker

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY

docker compose up --build
```

The service is then available at `http://localhost:8000`. Uploaded files,
the Chroma index, and the SQLite database (jobs + chat history) persist in
`./data` on the host.

## Deploying to a Cloud Host

This service writes to disk (SQLite for job tracking and chat history,
Chroma for the vector index), so where that data lives depends on the
host and plan you pick.

### Render

1. Push this repo to GitHub.
2. In the Render dashboard: **New → Blueprint**, point it at the repo.
   `render.yaml` (included) defines the web service on the free plan and
   reads `Dockerfile` to build.
3. Set `OPENROUTER_API_KEY` as a secret env var in the Render dashboard
   (not committed to the repo).
4. Deploy. Render builds the Dockerfile and gives you a public URL.

**Free-tier tradeoff:** Render's free plan doesn't support persistent
disks. `data/` lives on the container's local filesystem, which is wiped
on every redeploy **and** every time the service spins back up after
going idle (free services sleep after ~15 minutes of inactivity). In
practice: uploads, the index, and chat sessions survive while the service
is actively being used, but expect to re-upload documents and lose old
conversations after a period of inactivity or a redeploy. Reasonable for
a portfolio demo; upgrade to a paid plan and add a `disk:` block back to
`render.yaml` if you need real persistence.

### Fly.io

1. Install the `flyctl` CLI and run `fly auth login`.
2. From the project root: `fly launch` — it'll detect the `Dockerfile`
   and the included `fly.toml`. Say no to a Postgres/Redis add-on
   (not needed).
3. Create and attach a volume for persistent data:
   
   ```bash
   fly volumes create data --size 1 --region <your-region>
   ```
   
   (`fly.toml` already declares the mount at `/code/data`.)
4. Set your API key as a secret (never put it in `fly.toml`):
   
   ```bash
   fly secrets set OPENROUTER_API_KEY=sk-or-v1-...
   ```
5. Deploy:
   
   ```bash
   fly deploy
   ```

Fly's free allowance covers small always-on instances plus a small
volume, so uploads, the index, and chat history actually persist here
without upgrading.

### Either way

- Set `CORS_ORIGINS` in your platform's env vars to your actual frontend
  domain once you have one — leaving it as `*` is fine for testing, not
  for a public deployment.
- The `/health` endpoint is what most platforms will hit to confirm the
  service is alive — already wired up.
- There's no auth on `/upload` or `/chat` — anyone with the URL can use
  (and pay for, via your API key) the service. Fine for a private demo
  link, not for anything public-facing without adding an API key check.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

Tests mock all external LLM/embedding calls, so no API key or network access
is required to run the suite.

## Linting

```bash
ruff check app tests
```

## Configuration

See `.env.example` for all available environment variables — model
selection, chunk size/overlap, upload limits, CORS origins, re-ranking, and
chat history settings.

## Known Limitations / Next Steps

- Job store, chat history, and vector store (Chroma, local persistence)
  are all single-instance — fine for one deployment, but you'd want
  Postgres/Redis and a managed vector DB before running multiple app
  replicas.
- No auth on the API endpoints yet — add an API key or OAuth layer before
  exposing this publicly.
- `/chat` is not streamed — the client waits for the full LLM response.
- Re-ranking and question-condensing each add one extra LLM call per chat
  turn — toggle `RERANK_ENABLED` / `CONDENSE_QUESTION_ENABLED` off if
  latency/cost matters more than the quality gain.
- Uploaded source files are never cleaned up after processing.
- No document-type-specific parsing yet (e.g. table extraction) — plain
  text extraction only.
- Chat history has no expiry/cleanup — sessions accumulate indefinitely.
