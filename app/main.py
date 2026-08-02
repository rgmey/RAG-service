# app/main.py
import logging
import shutil
import sqlite3
from pathlib import Path

from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _count_active_jobs(jobs_db_path: Path) -> int:
    """Returns how many jobs are still pending/processing in an existing
    jobs.db, without importing job_store (which would open a persistent
    handle before we've decided whether to wipe the directory it lives
    in). Returns 0 if the file or table doesn't exist yet — nothing to
    protect in that case."""
    if not jobs_db_path.exists():
        return 0

    try:
        conn = sqlite3.connect(jobs_db_path)
        try:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status IN ('pending', 'processing')"
            )
            return cursor.fetchone()[0]
        finally:
            conn.close()
    except sqlite3.OperationalError:
        # e.g. table doesn't exist yet in a fresh/empty db file
        return 0


# Local-dev convenience: wipe uploads/Chroma index/jobs/chat history so
# every server start begins from a clean slate. Runs BEFORE the routes
# import below, since that import chain is what opens the Chroma client
# and SQLite files — wiping after they're opened risks file-lock errors
# (especially on Windows). Off by default; never enable this in a real
# deployment (Docker/Render/Fly) or a restart will delete production data.
#
# Guarded against wiping while a background job (upload processing) is
# still in flight: with --reload, a file save mid-upload spawns a fresh
# process, and that process's reset would otherwise delete the very job
# record the still-running background task (in the old process) is about
# to write its result to — producing a confusing "Job not found" with no
# indication why. Skipping the wipe when jobs are active avoids that.
if settings.DEV_RESET_DATA_ON_START:
    data_dir = Path(settings.LOCAL_DATA_DIR)
    active_jobs = _count_active_jobs(Path(settings.JOBS_DB_PATH))

    if active_jobs > 0:
        logger.warning(
            "DEV_RESET_DATA_ON_START=true but %d job(s) are still pending/processing "
            "in '%s' — skipping the reset this run so the in-flight upload isn't lost. "
            "It'll wipe normally on the next start once nothing is in flight.",
            active_jobs,
            data_dir,
        )
    else:
        if data_dir.exists():
            shutil.rmtree(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.warning(
            "DEV_RESET_DATA_ON_START=true — wiped '%s' (uploads, vector index, "
            "jobs, chat history). Disable this in .env once you don't want a "
            "fresh slate on every restart.",
            data_dir,
        )

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from app.api.routes import router  # noqa: E402

app = FastAPI(title="RAG Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health():
    """Liveness/readiness probe for load balancers and orchestrators."""
    return {"status": "ok"}


# Mounted last and at "/" so it only catches requests that didn't match
# an API route above (e.g. "/", "/index.html") — API paths like /chat
# and /upload are matched first since they were registered earlier.
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
