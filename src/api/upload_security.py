"""Path-boundary helpers for uploaded and server-local files."""

from __future__ import annotations

import re
import unicodedata
import uuid
from pathlib import Path, PurePosixPath
from typing import Iterable


_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_UNSAFE_FILENAME_CHARS = re.compile(r"[^\w.()\-\u4e00-\u9fff]+", re.UNICODE)


def sanitize_upload_filename(raw_name: str, max_chars: int = 200) -> str:
    if not raw_name or _CONTROL_CHARS.search(raw_name):
        raise ValueError("Invalid upload filename")
    normalized = unicodedata.normalize("NFKC", raw_name).replace("\\", "/")
    base_name = PurePosixPath(normalized).name.strip(" .")
    if base_name in {"", ".", ".."}:
        raise ValueError("Invalid upload filename")
    safe_name = _UNSAFE_FILENAME_CHARS.sub("_", base_name).strip(" ._")
    if not safe_name:
        raise ValueError("Invalid upload filename")
    if len(safe_name) > max_chars:
        suffix = Path(safe_name).suffix[:20]
        safe_name = safe_name[: max_chars - len(suffix)].rstrip(" ._") + suffix
    return safe_name


def upload_destination(upload_dir: Path | str, raw_name: str) -> Path:
    root = Path(upload_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_upload_filename(raw_name)
    candidate = root / safe_name
    if candidate.exists():
        suffix = candidate.suffix
        stem = candidate.name[: -len(suffix)] if suffix else candidate.name
        candidate = root / f"{stem}_{uuid.uuid4().hex[:10]}{suffix}"
    resolved = candidate.resolve(strict=False)
    if resolved.parent != root:
        raise ValueError("Upload path escapes the configured upload directory")
    return resolved


def resolve_local_import_paths(
    paths: Iterable[str], allowed_roots: Iterable[str], enabled: bool
) -> list[str]:
    if not enabled:
        raise PermissionError("Server-local path imports are disabled")
    configured_roots = [Path(root) for root in allowed_roots if root]
    if any(not root.is_absolute() for root in configured_roots):
        raise PermissionError("Server-local import roots must be absolute paths")
    roots = [root.resolve(strict=True) for root in configured_roots]
    if not roots:
        raise PermissionError("No server-local import roots are configured")

    resolved_paths: list[str] = []
    for raw_path in paths:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            raise ValueError(f"Local import path must be absolute: {raw_path}")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"Local import path does not exist: {raw_path}") from exc
        if not resolved.is_file():
            raise ValueError(f"Local import path is not a file: {raw_path}")
        if not any(resolved.is_relative_to(root) for root in roots):
            raise PermissionError(f"Local import path is outside allowed roots: {raw_path}")
        resolved_paths.append(str(resolved))
    if not resolved_paths:
        raise ValueError("No local import paths were provided")
    return resolved_paths
