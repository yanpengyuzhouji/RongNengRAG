"""Streaming request-size enforcement shared by reverse proxies."""

from __future__ import annotations


class StreamLimitExceeded(ValueError):
    pass


async def limited_stream(source, max_bytes: int):
    """Yield chunks without buffering and stop before exceeding ``max_bytes``."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    total = 0
    async for chunk in source:
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise StreamLimitExceeded(
                f"Request body exceeds the {max_bytes}-byte proxy limit"
            )
        yield chunk
