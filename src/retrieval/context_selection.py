"""Deterministic chunk-level context selection."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List


def select_context_chunks(
    results: Iterable[object], max_chunks: int, max_chunks_per_file: int
) -> List[object]:
    if max_chunks <= 0 or max_chunks_per_file <= 0:
        return []
    selected = []
    seen_chunks = set()
    per_file = defaultdict(int)
    for result in results:
        chunk_id = getattr(result, "chunk_id", "")
        file_path = getattr(result, "file_path", "")
        page_num = getattr(result, "page_num", 0)
        text = getattr(result, "text", "")
        chunk_key = (
            ("id", chunk_id)
            if chunk_id
            else ("content", file_path, page_num, text)
        )
        if chunk_key in seen_chunks:
            continue
        file_key = file_path or ("unscoped", chunk_key)
        if per_file[file_key] >= max_chunks_per_file:
            continue
        seen_chunks.add(chunk_key)
        per_file[file_key] += 1
        selected.append(result)
        if len(selected) >= max_chunks:
            break
    return selected
