"""Filename-series recognition used by focused document retrieval."""

from __future__ import annotations

import re
from typing import Optional


_CHAPTER_RE = re.compile(
    r"第[〇零一二三四五六七八九十百千万两\d]+(章节|章|节|部分|篇)"
)


def extract_series_key(filename: str) -> Optional[str]:
    if "会议材料之" in filename:
        return "会议材料之"
    match = _CHAPTER_RE.search(filename)
    if match:
        # Normalize the varying ordinal so 第1章 and 第十二章 share a key,
        # while chapter/section/part series remain distinct.
        return f"第*{match.group(1)}"
    return None
