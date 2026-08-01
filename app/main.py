# app/main.py
import logging
import shutil
from pathlib import Path

from app.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Local-dev convenience: wipe uploads/Chroma index/jobs/chat history so
# every server start begins from a clean slate. Runs BEFORE the routes
# import below, since that import chain is what opens the Chroma client
# and SQLite files — wiping after they're opened risks file-lock errors
# (especially on Windows). Off by default; never enable this in a real
# deployment (Docker/Render/Fly) or a restart will delete production data.
if settings.DEV_RESET_DATA_ON_START:
    data_dir = Path(settings.LOCAL_DATA_DIR)
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
