import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def _import_app_main_in_subprocess(data_dir: Path, reset_on_start: bool) -> None:
    """Runs `import app.main` in a fresh interpreter, the same way uvicorn's
    own import does — needed because the reset logic runs at import time,
    which we can't exercise by re-importing inside the same test process
    (the module would already be cached from test_api.py)."""
    env = {
        "OPENROUTER_API_KEY": "test-key",
        "LOCAL_DATA_DIR": str(data_dir),
        "CHROMA_PERSIST_DIR": str(data_dir / "chroma"),
        "JOBS_DB_PATH": str(data_dir / "jobs.db"),
        "DEV_RESET_DATA_ON_START": "true" if reset_on_start else "false",
        "PATH": os.environ.get("PATH", ""),
    }
    result = subprocess.run(
        [sys.executable, "-c", "import app.main"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_dev_reset_wipes_existing_data_dir():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        data_dir.mkdir()
        stale_file = data_dir / "old_upload.pdf"
        stale_file.write_text("stale data from a previous run")

        _import_app_main_in_subprocess(data_dir, reset_on_start=True)

        assert not stale_file.exists()
        assert data_dir.exists()  # wiped, then recreated empty


def test_dev_reset_disabled_by_default_leaves_data_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        data_dir.mkdir()
        existing_file = data_dir / "old_upload.pdf"
        existing_file.write_text("data from a previous run")

        _import_app_main_in_subprocess(data_dir, reset_on_start=False)

        assert existing_file.exists()
        assert existing_file.read_text() == "data from a previous run"


def test_dev_reset_skips_wipe_when_a_job_is_still_processing():
    """Regression test for the exact bug reported: a --reload restart
    mid-upload used to wipe the job record a background task (from the
    now-dead old process) was still about to write its result to,
    producing an unexplained 'job not found'. The reset must skip
    itself, not the job, when this happens."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        data_dir.mkdir()

        jobs_db_path = data_dir / "jobs.db"
        conn = sqlite3.connect(jobs_db_path)
        conn.execute(
            "CREATE TABLE jobs (job_id TEXT PRIMARY KEY, file_path TEXT, status TEXT, "
            "chunk_count INTEGER, error TEXT, created_at REAL, updated_at REAL)"
        )
        conn.execute(
            "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("in-flight-job", "/tmp/fake.pdf", "processing", None, None, 0.0, 0.0),
        )
        conn.commit()
        conn.close()

        stale_marker = data_dir / "should_survive.txt"
        stale_marker.write_text("this job is still processing, don't delete me")

        _import_app_main_in_subprocess(data_dir, reset_on_start=True)

        # The wipe should have been skipped entirely because of the active job.
        assert stale_marker.exists()
        assert stale_marker.read_text() == "this job is still processing, don't delete me"
