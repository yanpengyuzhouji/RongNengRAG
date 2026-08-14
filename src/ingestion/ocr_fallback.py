"""Safe selection between extracted PDF text and optional OCR output."""

from __future__ import annotations

from typing import Callable, Tuple


def choose_page_text(
    original_text: str | None,
    ocr_text: str | None,
    is_garbage: Callable[[str], bool],
) -> Tuple[str, str]:
    """Return ``(text, source)``; invalid OCR never discards usable source text."""
    original = (original_text or "").strip()
    ocr = (ocr_text or "").strip()
    if ocr:
        try:
            if not is_garbage(ocr):
                return ocr, "ocr"
        except Exception:
            # A detector failure is not evidence that source text is unusable.
            pass
    if original:
        return original, "pdf_text_fallback"
    return "", "empty"
