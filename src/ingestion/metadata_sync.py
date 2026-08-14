"""Cross-store metadata synchronization helpers.

Milvus Lite and the registry SQLite database cannot share a native transaction.
This module keeps the operation ordered and compensatable: vector rows are
snapshotted and updated first, the registry is committed second, and the vector
snapshot is restored if the registry commit fails.
"""

from __future__ import annotations

from typing import Callable, Dict, Mapping


METADATA_LIMITS = {"domain": 32, "category": 64, "doc_number": 256}


def normalize_metadata_updates(updates: Mapping[str, object]) -> Dict[str, str]:
    """Return validated, schema-bounded user-editable metadata."""
    normalized: Dict[str, str] = {}
    for field, max_length in METADATA_LIMITS.items():
        if field not in updates:
            continue
        value = str(updates[field] or "").strip()
        if "\x00" in value:
            raise ValueError(f"{field} contains a NUL character")
        if len(value) > max_length:
            raise ValueError(f"{field} exceeds {max_length} characters")
        normalized[field] = value
    return normalized


def inherit_registry_metadata(
    detected: Mapping[str, object],
    registry: Mapping[str, object] | None,
    *,
    explicit_domain: str | None = None,
    explicit_category: str | None = None,
) -> dict:
    """Preserve reviewed registry values when rebuilding a file index."""
    result = dict(detected)
    registry = registry or {}
    for field in METADATA_LIMITS:
        value = registry.get(field)
        if value not in (None, ""):
            result[field] = value
    if explicit_domain is not None:
        result["domain"] = explicit_domain
    if explicit_category is not None:
        result["category"] = explicit_category
    return result


def synchronize_metadata(
    *,
    file_hash: str,
    updates: Mapping[str, str],
    expected_chunks: int,
    store,
    commit_registry: Callable[[Mapping[str, str]], None],
) -> int:
    """Synchronize Milvus rows and SQLite with compensating rollback.

    ``store`` must implement ``replace_file_metadata`` and
    ``restore_file_snapshot``.  The returned integer is the number of vector
    rows verified with the new metadata.
    """
    snapshot = store.replace_file_metadata(
        file_hash, dict(updates), expected_count=expected_chunks
    )
    updated_chunks = len(snapshot)
    try:
        commit_registry(updates)
    except BaseException as registry_error:
        if snapshot:
            try:
                store.restore_file_snapshot(snapshot)
            except BaseException as rollback_error:
                raise RuntimeError(
                    "Registry update failed and Milvus metadata rollback also failed"
                ) from rollback_error
        raise registry_error
    return updated_chunks
