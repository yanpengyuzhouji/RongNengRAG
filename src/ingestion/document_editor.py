"""Pure helpers for editing persisted OCR layout documents.

The layout cache is the visual source of truth.  These helpers deliberately do
not touch Milvus or the filesystem, which makes edit validation and text
reconstruction deterministic and independently testable.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import re
from html.parser import HTMLParser
from typing import Dict, Iterable, List, Tuple


# Keep this set in sync with the renderer.  Some OCR/PDF layouts label
# non-text regions as header/footer images, photos, or seals.  If one of these
# labels is omitted here, the preview can hide the image while its covered OCR
# blocks remain indexable after save.
VISUAL_BLOCK_KINDS = {
    "chart",
    "image",
    "figure",
    "diagram",
    "picture",
    "photo",
    "seal",
    "header_image",
    "footer_image",
}
MAX_EDIT_OPERATIONS = 5000
MAX_BLOCK_CONTENT = 65535
_SAFE_DATA_IMAGE_RE = re.compile(
    r"data:image/(?:png|jpeg|gif|webp);base64,[A-Za-z0-9+/=]+",
    flags=re.IGNORECASE,
)
_INLINE_DATA_IMAGE_TAG_RE = re.compile(
    r'<img\b[^>]*\bsrc\s*=\s*"data:image/(?:png|jpeg|gif|webp);base64,[A-Za-z0-9+/=]+"[^>]*>',
    flags=re.IGNORECASE,
)


class LayoutEditError(ValueError):
    """The requested layout mutation is invalid."""


class LayoutRevisionConflict(LayoutEditError):
    """The client edited an older layout revision."""


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"br", "p", "div", "tr", "li", "table"}:
            self.parts.append("\n")
        elif tag.lower() in {"td", "th"}:
            self.parts.append("\t")

    def handle_endtag(self, tag):
        if tag.lower() in {"p", "div", "tr", "li", "table"}:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def layout_revision(layout_cache) -> str:
    """Return a stable optimistic-concurrency token for a layout cache."""
    canonical = json.dumps(
        layout_cache,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _block_kind(block: dict) -> str:
    return str(
        block.get("block_type")
        or block.get("block_label")
        or block.get("type")
        or "text"
    ).strip().lower()


def _block_box(block: dict):
    box = (
        block.get("bbox")
        or block.get("block_bbox")
        or block.get("block_box")
        or block.get("box")
    )
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in box[:4]]
    except (TypeError, ValueError):
        return None
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _page_entries(layout_cache) -> List[Tuple[int, List[dict]]]:
    if isinstance(layout_cache, list):
        return [(1, layout_cache)]
    if not isinstance(layout_cache, dict):
        raise LayoutEditError("版面缓存格式无效")
    entries = []
    for raw_page, blocks in layout_cache.items():
        try:
            page_num = int(raw_page) + 1
        except (TypeError, ValueError) as exc:
            raise LayoutEditError(f"版面页码无效: {raw_page}") from exc
        if not isinstance(blocks, list):
            raise LayoutEditError(f"第 {page_num} 页版面块格式无效")
        entries.append((page_num, blocks))
    return sorted(entries, key=lambda item: item[0])


def _sanitize_content(value, content_format: str) -> str:
    content = str(value or "").replace("\x00", "").replace("\r\n", "\n")
    if len(content) > MAX_BLOCK_CONTENT:
        raise LayoutEditError(f"单个版面块不能超过 {MAX_BLOCK_CONTENT} 个字符")
    if content_format == "html":
        # Keep passive table markup while removing active/external content.
        content = re.sub(
            r"<(script|style|iframe|object|embed|form|link)\b[^>]*>.*?</\1\s*>",
            "",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        # Editor-only wrappers/buttons are added around inline table images in
        # the iframe. Never persist those controls in the layout cache.
        content = re.sub(
            r"<button\b[^>]*\blayout-delete-inline-image\b[^>]*>.*?</button\s*>",
            "",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        content = re.sub(
            r"</?span\b[^>]*\blayout-inline-image\b[^>]*>",
            "",
            content,
            flags=re.IGNORECASE,
        )
        content = re.sub(r"\s+on\w+\s*=\s*(['\"]).*?\1", "", content, flags=re.I | re.S)
        def preserve_safe_image_src(match):
            value = match.group("double") or match.group("single") or ""
            return f' src="{value}"' if _SAFE_DATA_IMAGE_RE.fullmatch(value) else ""

        content = re.sub(
            r"\s+src\s*=\s*(?:\"(?P<double>[^\"]*)\"|'(?P<single>[^']*)')",
            preserve_safe_image_src,
            content,
            flags=re.I | re.S,
        )
        content = re.sub(r"\s+(href|action|formaction)\s*=\s*(['\"]).*?\2", "", content, flags=re.I | re.S)
    else:
        # Browser innerText may contain non-breaking spaces from rendered OCR.
        content = html.unescape(content).replace("\u00a0", " ")
    return content.strip()


def _covered_text_indices(blocks: List[dict], visual_index: int) -> set:
    visual_box = _block_box(blocks[visual_index])
    if not visual_box:
        return set()
    vx1, vy1, vx2, vy2 = visual_box
    covered = set()
    for index, candidate in enumerate(blocks):
        if index == visual_index or not isinstance(candidate, dict):
            continue
        kind = _block_kind(candidate)
        if kind in VISUAL_BLOCK_KINDS or "title" in kind or "heading" in kind:
            continue
        box = _block_box(candidate)
        if not box:
            continue
        x1, y1, x2, y2 = box
        area = max(1.0, (x2 - x1) * (y2 - y1))
        intersection = max(0.0, min(vx2, x2) - max(vx1, x1)) * max(
            0.0, min(vy2, y2) - max(vy1, y1)
        )
        coverage = intersection / area
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        if coverage >= 0.80 or (
            coverage >= 0.55
            and vx1 <= center_x <= vx2
            and vy1 <= center_y <= vy2
        ):
            covered.add(index)
    return covered


def apply_layout_edits(layout_cache, edits: Iterable[dict]):
    """Validate and apply block edits, returning ``(new_cache, audit_rows)``."""
    operations = list(edits or [])
    if not operations:
        raise LayoutEditError("没有可保存的修改")
    if len(operations) > MAX_EDIT_OPERATIONS:
        raise LayoutEditError(f"一次最多保存 {MAX_EDIT_OPERATIONS} 处修改")

    updated = copy.deepcopy(layout_cache)
    pages: Dict[int, List[dict]] = dict(_page_entries(updated))
    grouped: Dict[int, List[dict]] = {}
    seen = set()
    for operation in operations:
        if not isinstance(operation, dict):
            raise LayoutEditError("修改项格式无效")
        try:
            page_num = int(operation.get("page_num"))
            block_index = int(operation.get("block_index"))
        except (TypeError, ValueError) as exc:
            raise LayoutEditError("修改项缺少有效页码或版面块序号") from exc
        op = str(operation.get("op") or "update").lower()
        if op not in {"update", "delete", "delete_table_image"}:
            raise LayoutEditError(f"不支持的修改操作: {op}")
        image_index = operation.get("image_index")
        if op == "delete_table_image":
            try:
                image_index = int(image_index)
            except (TypeError, ValueError) as exc:
                raise LayoutEditError("删除表格图片缺少有效图片序号") from exc
            if image_index < 0:
                raise LayoutEditError("删除表格图片序号无效")
            key = (page_num, block_index, op, image_index)
        else:
            key = (page_num, block_index)
        if key in seen:
            raise LayoutEditError("同一版面块不能在一次保存中重复修改")
        seen.add(key)
        if page_num not in pages or block_index < 0 or block_index >= len(pages[page_num]):
            raise LayoutEditError(f"第 {page_num} 页版面块 {block_index} 不存在")
        grouped.setdefault(page_num, []).append({
            **operation, "op": op, "block_index": block_index, "image_index": image_index,
        })

    audit_rows = []
    for page_num, page_operations in grouped.items():
        blocks = pages[page_num]
        delete_indices = set()
        # Delete higher inline-image indexes first, otherwise removing image 0
        # would shift image 1 to index 0 in the same table block.
        ordered_operations = sorted(
            page_operations,
            key=lambda operation: (
                operation["block_index"],
                0 if operation["op"] == "delete_table_image" else 1,
                -int(operation.get("image_index") or 0),
            ),
        )
        for operation in ordered_operations:
            index = operation["block_index"]
            block = blocks[index]
            if not isinstance(block, dict):
                raise LayoutEditError(f"第 {page_num} 页版面块 {index} 格式无效")
            before = str(block.get("block_content") or "")
            if operation["op"] == "delete_table_image":
                image_index = operation["image_index"]
                matches = list(_INLINE_DATA_IMAGE_TAG_RE.finditer(before))
                if image_index >= len(matches):
                    raise LayoutEditError(f"第 {page_num} 页版面块 {index} 的图片序号不存在")
                target = matches[image_index]
                block["block_content"] = before[:target.start()] + before[target.end():]
                audit_rows.append({
                    "page_num": page_num,
                    "block_index": index,
                    "op": "delete_table_image",
                    "kind": _block_kind(block),
                    "before": "[表格内图片]",
                    "after": "",
                })
                continue
            if operation["op"] == "delete":
                delete_indices.add(index)
                if _block_kind(block) in VISUAL_BLOCK_KINDS:
                    delete_indices.update(_covered_text_indices(blocks, index))
                audit_rows.append({
                    "page_num": page_num,
                    "block_index": index,
                    "op": "delete",
                    "kind": _block_kind(block),
                    "before": before or "[图片]",
                    "after": "",
                })
                continue
            after = _sanitize_content(
                operation.get("content", ""),
                str(operation.get("content_format") or "text").lower(),
            )
            block["block_content"] = after
            audit_rows.append({
                "page_num": page_num,
                "block_index": index,
                "op": "update",
                "kind": _block_kind(block),
                "before": before,
                "after": after,
            })
        for index in sorted(delete_indices, reverse=True):
            del blocks[index]
    return updated, audit_rows


def _html_to_text(value: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(value)
        parser.close()
        text = "".join(parser.parts)
    except Exception:
        text = re.sub(r"<[^>]+>", " ", value)
    text = html.unescape(text).replace("\u00a0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def layout_page_texts(layout_cache) -> List[Tuple[int, str]]:
    """Rebuild page text from the same edited blocks used by preview."""
    page_texts = []
    for page_num, blocks in _page_entries(layout_cache):
        ordered = sorted(
            (block for block in blocks if isinstance(block, dict)),
            key=lambda block: (
                (_block_box(block) or (0, 0, 0, 0))[1],
                (_block_box(block) or (0, 0, 0, 0))[0],
            ),
        )
        parts = []
        for block in ordered:
            if _block_kind(block) in VISUAL_BLOCK_KINDS:
                continue
            content = str(block.get("block_content") or "").strip()
            if not content:
                continue
            parts.append(_html_to_text(content) if "<" in content and ">" in content else content)
        page_texts.append((page_num, "\n\n".join(part for part in parts if part.strip())))
    return page_texts


def layout_text_fingerprints(layout_cache) -> Dict[int, str]:
    """Return a stable per-page fingerprint of text that participates in search.

    Visual assets, HTML structure and whitespace-only edits intentionally do
    not change this result. Callers can therefore skip vector synchronization
    when a save only changes preview presentation.
    """
    return {
        page_num: hashlib.sha256(
            re.sub(r"\s+", " ", text or "").strip().encode("utf-8")
        ).hexdigest()
        for page_num, text in layout_page_texts(layout_cache)
    }


def count_visual_assets(layout_cache) -> int:
    count = 0
    for _, blocks in _page_entries(layout_cache):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if (
                _block_kind(block) in VISUAL_BLOCK_KINDS
                and str(block.get("visual_data_uri") or "").startswith("data:image/")
            ):
                count += 1
            # Pipeline table cells can contain locally derived data-image URIs
            # instead of separate image/chart layout blocks.
            count += len(re.findall(
                r"data:image/(?:png|jpeg|gif|webp);base64,[A-Za-z0-9+/=]+",
                str(block.get("block_content") or ""),
                flags=re.IGNORECASE,
            ))
    return count
