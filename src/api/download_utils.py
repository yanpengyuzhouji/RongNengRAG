"""Helpers for preserving the original file format during downloads."""

import mimetypes
from pathlib import Path, PurePosixPath


# Keep the common knowledge-base formats stable across operating systems.
# ``mimetypes`` differs between Windows and Linux, and some office formats
# are not registered on minimal server images.
_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".wps": "application/vnd.ms-works",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ofd": "application/ofd",
    ".ceb": "application/x-ceb",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}


def download_filename(file_name: str = "", path: str = "") -> str:
    """Return a safe basename for ``Content-Disposition`` and browser save."""
    raw = str(file_name or path or "").replace("\\", "/").strip()
    name = PurePosixPath(raw).name.strip(" .")
    return name or "download"


def download_media_type(file_name: str = "", path: str = "") -> str:
    """Resolve a deterministic MIME type from the original extension."""
    name = download_filename(file_name, path)
    suffix = Path(name).suffix.lower()
    if not suffix and path:
        suffix = Path(str(path)).suffix.lower()
    return _MEDIA_TYPES.get(suffix) or mimetypes.guess_type(name)[0] or "application/octet-stream"
