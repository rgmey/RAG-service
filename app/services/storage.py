# app/services/storage.py
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.core.config import settings

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _validate_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Allowed: {sorted(settings.ALLOWED_UPLOAD_EXTENSIONS)}"
            ),
        )
    return suffix


async def save_file(file: UploadFile) -> str:
    """Validates and saves an uploaded file, returning its path on disk.

    Uses a server-generated UUID for the filename — the original filename
    is never used to build a filesystem path, which rules out path
    traversal via a malicious filename.
    """
    suffix = _validate_extension(file.filename or "")

    contents = await file.read()
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds the {settings.MAX_UPLOAD_MB}MB upload limit",
        )

    dest = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    dest.write_bytes(contents)

    return str(dest)
