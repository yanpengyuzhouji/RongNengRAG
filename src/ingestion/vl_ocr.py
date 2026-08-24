"""
PaddleOCR-VL 结构化识别客户端 — 调用外部 OpenAI 兼容的 VL 模型服务
将 PDF 页面渲染为图片 → base64 → POST /v1/chat/completions → 返回结构化文本
支持阅读顺序、markdown 表格、标题层级 (PaddleOCR-VL 1.x 特性)
"""

import base64
import html as html_lib
import io
import re
import time
from typing import Dict, List, Optional

import fitz  # PyMuPDF

from ingestion.document_editor import VISUAL_BLOCK_KINDS


# 识别提示词: 要求保持阅读顺序 + 结构化输出
_VL_PROMPT = (
    "识别图中全部文字，保持阅读顺序输出；"
    "表格转为 markdown 表格；保留标题层级；"
    "数学公式必须保留为 LaTeX：行内公式使用 $...$，独立公式使用 $$...$$；"
    "保留下标、上标、分数、根号、希腊字母、运算符和单位，不要将公式改写成近似普通文字；"
    "只输出识别到的内容，不要额外解释。"
)

# 小模型幻觉特征: 生成自我解释/占位性废话 (而非实际识别内容)
_ILLUSION_PHRASES = (
    "无法直接转为",
    "无法直接转写",
    "根据上述情况",
    "所以，根据",
    "在Markdown中添加Markdown",
    "以方便后续分析",
    "无法直接转换为",
)

# 重复性检测: 连续相同的表格单元 (如 "名称 | 100 | 名称 | 100 ...")
_REPEAT_CELL_RE = None  # 惰性编译


def _number(value, default=10**9):
    """Convert pipeline numeric fields, including JSON null, safely."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _block_xy(block: dict):
    """Extract the top-left coordinate from common Paddle layout schemas."""
    for key in ("bbox", "block_bbox", "block_box", "box", "coordinate", "coordinates"):
        value = block.get(key)
        if isinstance(value, dict):
            x, y = value.get("x", value.get("left")), value.get("y", value.get("top"))
            if x is not None and y is not None:
                return (_number(y), _number(x))
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            # Polygon/list formats are flattened and use min x/min y.
            nums = [_number(v) for v in value]
            if all(v < 10**9 for v in nums):
                return (min(nums[1::2] or nums), min(nums[0::2] or nums))
    return None


def _canonical_block(text: str) -> str:
    """Fingerprint text while ignoring HTML/Markdown whitespace noise."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip().casefold()


def _clean_fragmentary_html(text: str) -> str:
    """Drop orphan table-cell fragments while preserving complete tables."""
    if not text:
        return text
    tables = []

    def keep_table(match):
        tables.append(match.group(0))
        return f"\u0000TABLE_{len(tables) - 1}\u0000"

    masked = re.sub(r"<table\b[\s\S]*?</table\s*>", keep_table, text, flags=re.I)
    # Pipeline occasionally emits a suffix beginning with <td> and ending at
    # </table>, without the opening <table>. It is not renderable and often
    # contains concatenated values such as `415>50, ...`; discard the fragment.
    masked = re.sub(r"<td\b[\s\S]*?</table\s*>", "", masked, flags=re.I)
    # Remove any remaining standalone row/cell tags, retaining surrounding
    # plain text and Markdown tables.
    masked = re.sub(r"</?(?:tbody|thead|tfoot|tr|td|th)\b[^>]*>", "", masked, flags=re.I)
    # Some pipeline versions place a broken row inside a table and concatenate
    # numeric cells (e.g. `415>50, ≤100335300175`). This is not a valid cell
    # value; remove only the unmistakable glued-number pattern.
    masked = re.sub(
        r"\d{2,}>\d+\s*,\s*≤\d{9,}",
        "",
        masked,
    )
    for idx, table in enumerate(tables):
        masked = masked.replace(f"\u0000TABLE_{idx}\u0000", table)
    return masked.strip()


def _dedupe_blocks(values):
    seen = set()
    result = []
    for value in values:
        text = _clean_fragmentary_html(str(value or "").strip())
        key = _canonical_block(text)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _dedupe_markdown(text: str) -> str:
    """Remove repeated OCR paragraphs/tables without changing cell content."""
    parts = re.split(r"\n\s*\n", text.strip())
    return "\n\n".join(_dedupe_blocks(parts))


def _sanitize_layout_content(content: str) -> str:
    """Remove active/external HTML before putting OCR content in an iframe.

    Layout preview needs to keep passive table markup, but the preview iframe
    now runs MathJax.  Do not let OCR-returned HTML turn into arbitrary script
    or external-resource input when scripts are enabled for formula typesetting.
    """
    if not content:
        return ""
    content = re.sub(
        r"<\s*(script|iframe|object|embed|link|style|form)\b[^>]*>[\s\S]*?"
        r"<\s*/\s*\1\s*>",
        "",
        content,
        flags=re.I,
    )
    content = re.sub(
        r"<\s*/?\s*(script|iframe|object|embed|link|style|form)\b[^>]*/?>",
        "",
        content,
        flags=re.I,
    )
    # Remove event handlers and URL-bearing attributes. Table structure
    # attributes such as rowspan/colspan remain intact. Data-image URIs are
    # produced locally from the uploaded page and are the sole safe exception.
    content = re.sub(
        r"\s+on[a-z]+\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)",
        "",
        content,
        flags=re.I,
    )
    def preserve_safe_image_src(match):
        value = (
            match.group("double")
            or match.group("single")
            or match.group("bare")
            or ""
        )
        if re.fullmatch(r"data:image/(?:png|jpeg|gif|webp);base64,[A-Za-z0-9+/=]+", value, re.I):
            return f' src="{value}"'
        return ""

    content = re.sub(
        r"\s+src\s*=\s*(?:\"(?P<double>[^\"]*)\"|'(?P<single>[^']*)'|(?P<bare>[^\s>]+))",
        preserve_safe_image_src,
        content,
        flags=re.I,
    )
    content = re.sub(
        r"\s+(?:href|action|formaction)\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+)",
        "",
        content,
        flags=re.I,
    )
    return content


def _normalize_ocr_escaped_newlines(text: str) -> str:
    """Turn pipeline's literal ``\\n`` markers into real line breaks.

    Paddle's HTML table output sometimes serializes line breaks as two visible
    characters (backslash + n).  Avoid rewriting LaTeX commands such as
    ``\\neq`` while normalizing the layout separators.
    """
    if not text or "\\n" not in text:
        return text

    latex_n_commands = (
        "eq", "e", "abla", "u", "ot", "otin", "mid", "exists",
        "eg", "ormal", "ewcommand",
    )

    def replace(match):
        remainder = text[match.end():]
        if any(
            re.match(rf"{re.escape(command)}(?![A-Za-z])", remainder)
            for command in latex_n_commands
        ):
            return match.group(0)
        return "\n"

    return re.sub(r"\\n", replace, text)


def _compact_layout_text(text: str) -> str:
    """Collapse OCR line-wraps in ordinary text blocks.

    A pipeline text block is already a visual region.  Preserving every OCR
    line break creates artificial vertical gaps and makes headings consume
    more space than the source page.  Tables are handled separately and keep
    their meaningful cell line breaks.
    """
    normalized = _normalize_ocr_escaped_newlines(text or "")
    return re.sub(r"\s+", " ", normalized).strip()


# The editor and renderer must classify exactly the same visual block kinds so
# deleting a visual block also removes any OCR text covered by its bbox.
_VISUAL_BLOCK_KINDS = VISUAL_BLOCK_KINDS
_MAX_VISUAL_ASSET_BYTES = 2 * 1024 * 1024
_PIPELINE_IMAGE_TAG_RE = re.compile(r"<img\b[^>]*>", flags=re.I)
_PIPELINE_IMAGE_BOX_RE = re.compile(
    r"(?:^|[\\/'\"=])imgs[\\/]img_in_image_box_"
    r"(?P<x1>\d+)_(?P<y1>\d+)_(?P<x2>\d+)_(?P<y2>\d+)\.(?:jpe?g|png)",
    flags=re.I,
)
_SAFE_DATA_IMAGE_RE = re.compile(
    r"data:image/(?:png|jpeg|gif|webp);base64,[A-Za-z0-9+/=]+",
    flags=re.I,
)


def _is_visual_block_kind(kind: str) -> bool:
    """Return whether a layout label represents a non-text visual region."""
    normalized = str(kind or "").strip().lower()
    return normalized in _VISUAL_BLOCK_KINDS


def _layout_block_box(block: dict):
    """Read a rectangular layout bbox from the common Pipeline fields."""
    if not isinstance(block, dict):
        return None
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
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _attach_visual_assets(image_bytes: bytes, blocks: list) -> list:
    """Crop visual layout blocks from the exact image sent to OCR.

    Pipeline chart/image blocks usually contain only a bbox and no textual
    content.  Embedding a bounded PNG data URI in the cached block keeps the
    existing ``srcdoc`` preview self-contained and avoids a second authenticated
    asset endpoint.  If Pillow or a crop fails, the original block is retained.
    """
    if not image_bytes or not isinstance(blocks, list):
        return list(blocks or [])
    if not any(
        _is_visual_block_kind(
            str(
                block.get("block_type")
                or block.get("block_label")
                or block.get("type")
                or "text"
            ).lower()
        )
        for block in blocks
        if isinstance(block, dict)
    ):
        return [dict(block) if isinstance(block, dict) else block for block in blocks]
    try:
        from PIL import Image

        source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        print(f"[vl-ocr] visual asset source unavailable: {exc}", flush=True)
        return [dict(block) if isinstance(block, dict) else block for block in blocks]

    width, height = source.size
    attached = []
    for block in blocks:
        if not isinstance(block, dict):
            attached.append(block)
            continue
        item = dict(block)
        kind = str(
            item.get("block_type")
            or item.get("block_label")
            or item.get("type")
            or "text"
        ).lower()
        if _is_visual_block_kind(kind) and not item.get("visual_data_uri"):
            box = _layout_block_box(item)
            if box:
                x1, y1, x2, y2 = box
                # Include a narrow border around charts so axes and figure
                # outlines are not clipped by integer rounding.
                pad = 3
                left = max(0, int(x1) - pad)
                top = max(0, int(y1) - pad)
                right = min(width, int(x2 + pad + 0.999))
                bottom = min(height, int(y2 + pad + 0.999))
                if right > left and bottom > top:
                    try:
                        output = io.BytesIO()
                        source.crop((left, top, right, bottom)).save(
                            output, format="PNG", optimize=True
                        )
                        payload = output.getvalue()
                        if len(payload) <= _MAX_VISUAL_ASSET_BYTES:
                            item["visual_data_uri"] = (
                                "data:image/png;base64,"
                                + base64.b64encode(payload).decode("ascii")
                            )
                            item["visual_asset_bbox"] = [left, top, right, bottom]
                        else:
                            print(
                                f"[vl-ocr] visual asset skipped ({len(payload)} bytes): {kind}",
                                flush=True,
                            )
                    except Exception as exc:
                        print(f"[vl-ocr] visual asset crop failed: {exc}", flush=True)
        attached.append(item)
    return attached


def _inline_pipeline_table_images(image_bytes: bytes, blocks: list) -> list:
    """Replace Pipeline's inaccessible table-image paths with safe image data URIs.

    PaddleOCR Pipeline detects photos inside tables but emits them as relative
    paths such as ``imgs/img_in_image_box_514_391_711_513.jpg``. Those files
    only exist inside the OCR container. The coordinates encode the crop on
    the submitted page image, so derive it locally for the layout cache only.
    """
    if not image_bytes or not isinstance(blocks, list):
        return list(blocks or [])
    if not any(
        isinstance(block, dict)
        and "img_in_image_box_" in str(block.get("block_content") or "")
        for block in blocks
    ):
        return [dict(block) if isinstance(block, dict) else block for block in blocks]
    try:
        from PIL import Image

        source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        print(f"[vl-ocr] inline table image source unavailable: {exc}", flush=True)
        return [dict(block) if isinstance(block, dict) else block for block in blocks]

    width, height = source.size
    crop_cache = {}

    def replace_tag(match):
        tag = match.group(0)
        location = _PIPELINE_IMAGE_BOX_RE.search(tag)
        if not location:
            return tag
        box = tuple(int(location.group(name)) for name in ("x1", "y1", "x2", "y2"))
        if box not in crop_cache:
            x1, y1, x2, y2 = box
            left, top = max(0, x1), max(0, y1)
            right, bottom = min(width, x2), min(height, y2)
            if right <= left or bottom <= top:
                crop_cache[box] = ""
            else:
                try:
                    output = io.BytesIO()
                    source.crop((left, top, right, bottom)).save(
                        output, format="PNG", optimize=True
                    )
                    payload = output.getvalue()
                    crop_cache[box] = (
                        "data:image/png;base64," + base64.b64encode(payload).decode("ascii")
                        if len(payload) <= _MAX_VISUAL_ASSET_BYTES else ""
                    )
                except Exception as exc:
                    print(f"[vl-ocr] inline table image crop failed: {exc}", flush=True)
                    crop_cache[box] = ""
        data_uri = crop_cache[box]
        return f'<img src="{data_uri}" alt="图片示例">' if data_uri else tag

    enriched = []
    for block in blocks:
        if not isinstance(block, dict):
            enriched.append(block)
            continue
        item = dict(block)
        content = str(item.get("block_content") or "")
        if "img_in_image_box_" in content:
            item["block_content"] = _PIPELINE_IMAGE_TAG_RE.sub(replace_tag, content)
        enriched.append(item)
    return enriched


def _decorate_editable_table_images(content: str) -> str:
    """Add per-image delete controls to already-sanitized table HTML."""
    if not content or "data:image/" not in content:
        return content

    image_index = 0

    def replace(match):
        nonlocal image_index
        tag = match.group(0)
        current_index = image_index
        image_index += 1
        return (
            '<span class="layout-inline-image">'
            f'{tag}<button type="button" class="layout-delete-inline-image" '
            f'data-inline-image-index="{current_index}" title="删除图片">删除图片</button></span>'
        )

    return re.sub(
        r'<img\b[^>]*\bsrc\s*=\s*"data:image/(?:png|jpeg|gif|webp);base64,[A-Za-z0-9+/=]+"[^>]*>',
        replace,
        content,
        flags=re.I,
    )


def _toc_depth(title: str) -> int:
    """Return a visual indentation depth for one contents entry."""
    value = str(title or "").strip()
    appendix = re.match(r"^附录\s*[A-Za-z一二三四五六七八九十百千万零〇0-9]+", value)
    if appendix:
        return 1
    numbered = re.match(r"^([A-Za-z]?\d+(?:\s*[.．]\s*\d+)*)\b", value)
    if not numbered:
        return 0
    number = re.sub(r"\s+", "", numbered.group(1))
    return number.count(".") + 1 + (1 if re.match(r"^[A-Za-z]", number) else 0)


def _render_toc_html(content: str) -> str:
    """Render OCR contents rows as title, leader and page-number columns.

    OCR commonly returns a contents page as ordinary text such as
    ``1 总则 ..... (1)``.  Rendering that string as a normal paragraph makes
    whitespace collapse and sends the leaders/page numbers out of alignment.
    Keep each source row, but use CSS flex columns so Chinese and English
    contents have the same stable visual structure.
    """
    rows = []
    normalized = _normalize_ocr_escaped_newlines(str(content or ""))
    for raw_line in normalized.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        match = re.match(
            r"^(.*?)\s*(?:\.{2,}|…{2,}|·{2,}|-{2,}|_{2,})\s*"
            r"[（(]?\s*(\d+)\s*[)）]?\s*$",
            line,
        )
        if not match:
            match = re.match(r"^(.*?)\s+[（(]\s*(\d+)\s*[)）]\s*$", line)
        title = (match.group(1) if match else line).strip()
        page = (match.group(2) if match else "").strip()
        if not title:
            continue
        indent = max(0, min(6, _toc_depth(title)) - 1) * 1.2
        title_html = html_lib.escape(title)
        page_html = html_lib.escape(page)
        if page:
            suffix = (
                '<span class="layout-toc-leader"></span>'
                f'<span class="layout-toc-page">{page_html}</span>'
            )
        else:
            suffix = ""
        rows.append(
            f'<div class="layout-toc-row" style="padding-left:{indent:.1f}em">'
            f'<span class="layout-toc-title">{title_html}</span>{suffix}</div>'
        )
    return '<div class="layout-toc">' + "".join(rows) + "</div>"


def _looks_like_toc(content: str) -> bool:
    """Detect a contents block even when OCR gives it a generic text label."""
    lines = [line.strip() for line in _normalize_ocr_escaped_newlines(str(content or "")).splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    marked = sum(
        bool(re.search(
            r"(?:\.{2,}|…{2,}|·{2,}|-{2,}|_{2,})\s*"
            r"[（(]?\s*\d+\s*[)）]?\s*$",
            line,
        ))
        for line in lines
    )
    return marked >= max(2, (len(lines) + 1) // 2)


def _outline_anchor_id(page_num: int, block: dict, block_index: int) -> str:
    """Build a stable anchor shared by the API outline and layout iframe."""
    raw_id = block.get("block_id", block_index) if isinstance(block, dict) else block_index
    raw_id = re.sub(r"[^A-Za-z0-9_-]+", "-", str(raw_id)).strip("-") or str(block_index)
    return f"outline-p{int(page_num or 1)}-b{raw_id}"


def _outline_title_level(text: str, kind: str = "text", allow_single_number: bool = False):
    """Return ``(title, level)`` for a likely section heading.

    OCR layout labels are useful but not perfectly consistent.  This parser
    combines the labels with common Chinese/technical numbering conventions,
    while rejecting ordinary numbered paragraphs that end in punctuation.
    """
    value = _normalize_ocr_escaped_newlines(str(text or ""))
    value = html_lib.unescape(re.sub(r"<[^>]+>", " ", value))
    first_line = re.split(r"\n", value, maxsplit=1)[0]
    first_line = re.sub(r"^\s*#{1,6}\s*", "", first_line)
    first_line = re.sub(r"\s+", " ", first_line).strip()
    if not first_line:
        return None
    # Table-of-contents leaders and page references are navigation data, not
    # document headings.  Dates/locations on a cover page are metadata too.
    if re.search(r"(?:\.{3,}|…{2,})", first_line) or re.search(r"[（(]\s*\d+\s*[)）]\s*$", first_line):
        return None
    if re.match(r"^(?:19|20)\d{2}(?:[-/.年]|\s+(?:北|上|广|北\s*京))", first_line):
        return None

    normalized_kind = str(kind or "text").lower()
    if normalized_kind in {"figure_title", "table_title", "header", "footer"}:
        return None

    is_labeled_heading = (
        "title" in normalized_kind
        or "heading" in normalized_kind
        or normalized_kind in {"paragraph_title", "doc_title", "document_title"}
    )
    # Do not return document titles yet: a pipeline may label a real chapter
    # (for example ``4 光缆线路敷设``) as ``doc_title``.  Let the numbering and
    # geometry rules classify it first, then fall back to level 0 below.

    chapter = re.match(
        r"^(第\s*[一二三四五六七八九十百千万零〇0-9]+\s*[章节篇])"
        r"\s*[、:：.．-]?\s*(.*)$",
        first_line,
    )
    if chapter:
        return (first_line, 1)

    appendix = re.match(
        r"^(附录\s*[A-Za-z一二三四五六七八九十百千万零〇0-9]+)"
        r"(?:\s*[、:：.．-]?\s*(.*))?$",
        first_line,
    )
    if appendix:
        suffix = appendix.group(2) or ""
        level = 1 + suffix.count(".") if suffix else 1
        return (first_line, level)

    numbered = re.match(
        r"^((?:[A-Za-z]\s*)?\d+(?:\s*[.．]\s*\d+)*)"
        r"(?:\s*[、:：.．-]\s*|\s+)(.+?)\s*$",
        first_line,
    )
    if numbered:
        number = re.sub(r"\s+", "", numbered.group(1))
        title = numbered.group(2).strip()
        depth = number.count(".") + 1 + (1 if re.match(r"^[A-Za-z]", number) else 0)
        # A bare ``6 标题`` is a chapter only when it looks like a title;
        # ordinary list items usually end in 。；, or are long sentences.
        if depth == 1 and not allow_single_number and not is_labeled_heading:
            return None
        if (
            not is_labeled_heading
            and (len(first_line) > 90 or (depth == 1 and re.search(r"[。；;，,:：]$", title)))
        ):
            return None
        return (first_line, depth)

    chinese_number = re.match(
        r"^([一二三四五六七八九十百千万零〇]+)[、.．:：]\s*(.+?)\s*$",
        first_line,
    )
    if chinese_number and (is_labeled_heading or allow_single_number):
        return (first_line, 1)

    alpha_number = re.match(
        r"^([A-Za-z]+\d+(?:[.．]\d+)+)\s+(.+?)\s*$",
        first_line,
    )
    if alpha_number and (is_labeled_heading or allow_single_number):
        return (first_line, alpha_number.group(1).count(".") + 2)

    if normalized_kind in {"doc_title", "document_title", "main_title", "title"}:
        return (first_line, 0)
    return None


def _outline_candidate(block: dict, page_width: float = 0, page_height: float = 0):
    if not isinstance(block, dict):
        return None
    kind = str(
        block.get("block_type")
        or block.get("block_label")
        or block.get("type")
        or "text"
    ).lower()
    content = block.get("block_content", "")
    if kind in {"content", "header", "footer", "number", "figure_title", "table_title"}:
        return None
    # Text blocks can contain headings such as ``6 蓄电池组配置`` even when
    # the OCR service did not assign a title label.  Keep the conservative
    # single-number rule in that case to avoid indexing numbered list items.
    parsed = _outline_title_level(
        content,
        kind,
        allow_single_number=kind in {"text", "paragraph", "content"},
    )
    if not parsed:
        return None
    # Unlabelled text is only promoted when its geometry resembles a heading:
    # short, centered, and not a footer-sized block.  Labeled title blocks are
    # trusted because the layout model has already classified them.
    is_labeled_heading = "title" in kind or "heading" in kind or kind in {"paragraph_title", "doc_title"}
    if not is_labeled_heading and page_width > 0:
        box = block.get("bbox") or block.get("block_bbox") or block.get("block_box") or block.get("box")
        try:
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
        except (TypeError, ValueError, IndexError):
            return None
        width_ratio = max(0.0, x2 - x1) / page_width
        center_delta = abs(((x1 + x2) / 2) - (page_width / 2)) / page_width
        raw_content = _normalize_ocr_escaped_newlines(str(content or ""))
        has_following_body = "\n" in raw_content
        if parsed[1] >= 2 and not has_following_body and len(parsed[0]) > 32:
            # A long, same-line numbered sentence is more likely a clause
            # than a clean section title; avoid putting its body in the TOC.
            return None
        if parsed[1] <= 1:
            if width_ratio > 0.60 or center_delta > 0.18:
                # Short appendix/definition labels such as ``D5 标题`` are
                # often left-indented rather than centered.
                if not (re.match(r"^[A-Za-z]\d+\b", parsed[0]) and len(parsed[0]) <= 40):
                    return None
        else:
            short_alpha_heading = bool(
                re.match(r"^[A-Za-z]\d+\b", parsed[0]) and len(parsed[0]) <= 40
            )
            punctuated = bool(re.search(r"[。；;，,:：]$", parsed[0]))
            if has_following_body and len(parsed[0]) <= 70:
                # Definition-style sections often put the title and its body
                # in one OCR block; the explicit line break is useful evidence.
                pass
            else:
                if width_ratio > 0.45 or (
                    center_delta > 0.25 and not (short_alpha_heading and not punctuated)
                ):
                    return None
                if (
                    punctuated
                    and (width_ratio > 0.35 or (short_alpha_heading and center_delta > 0.18))
                ):
                    # A compact, centered numbered line can be a subheading;
                    # a wider punctuated line is normally a requirement.
                    return None
        if page_height > 0 and y2 > page_height * 0.90:
            return None
    return parsed


def _iter_layout_pages(layout_pages):
    if isinstance(layout_pages, dict):
        entries = []
        for raw_page, blocks in layout_pages.items():
            try:
                page_num = int(raw_page) + 1
            except (TypeError, ValueError):
                continue
            entries.append((page_num, blocks if isinstance(blocks, list) else []))
        return sorted(entries, key=lambda item: item[0])
    if isinstance(layout_pages, list):
        return [(1, layout_pages)]
    return []


def extract_layout_outline(layout_pages) -> list:
    """Extract a hierarchical outline from cached OCR layout blocks."""
    outline = []
    seen = set()
    seen_numbers = set()
    for page_num, blocks in _iter_layout_pages(layout_pages):
        boxes = []
        for block in blocks:
            box = None
            if isinstance(block, dict):
                box = (
                    block.get("bbox")
                    or block.get("block_bbox")
                    or block.get("block_box")
                    or block.get("box")
                )
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                try:
                    boxes.append(tuple(float(v) for v in box[:4]))
                except (TypeError, ValueError):
                    pass
        if boxes:
            min_x = min(box[0] for box in boxes)
            page_width = max(box[2] for box in boxes) + max(0.0, min_x)
            page_height = max(box[3] for box in boxes)
        else:
            page_width = page_height = 0
        for block_index, block in enumerate(blocks):
            parsed = _outline_candidate(block, page_width, page_height)
            if not parsed:
                continue
            title, level = parsed
            if level == 0 and (
                re.search(r"Code for|Quality of electric|公告$", title, flags=re.I)
                or title.strip() in {"中华人民共和国标准", "中华人民共和国国家标准"}
            ):
                continue
            if level == 0 and any(item.get("level", 0) == 0 for item in outline):
                # Cover pages and appendices may repeat the document title;
                # keep the first meaningful root title in the navigation.
                continue
            key = re.sub(r"\s+", " ", title).strip().casefold()
            number_match = re.match(
                r"^\s*(附录\s*[A-Za-z0-9一二三四五六七八九十百千万零〇]+|"
                r"[A-Za-z]?\d+(?:\s*[.．]\s*\d+)*)",
                title,
                flags=re.I,
            )
            number_key = (
                re.sub(r"\s+", "", number_match.group(1)).replace("．", ".").casefold()
                if number_match else ""
            )
            if not key or key in seen:
                continue
            if level > 0 and number_key and number_key in seen_numbers:
                continue
            seen.add(key)
            if level > 0 and number_key:
                seen_numbers.add(number_key)
            outline.append({
                "id": _outline_anchor_id(page_num, block, block_index),
                "anchor": _outline_anchor_id(page_num, block, block_index),
                "title": title,
                "level": level,
                "page": page_num,
            })
    def sort_key(item):
        title = item["title"]
        if item.get("level", 0) == 0:
            return (0, (), item.get("page", 0))
        numeric = re.match(r"^\s*(\d+(?:\s*[.．]\s*\d+)*)", title)
        if numeric:
            parts = tuple(int(value) for value in re.findall(r"\d+", numeric.group(1)))
            return (1, parts, item.get("page", 0))
        appendix = re.match(r"^\s*附录\s*([A-Za-z0-9一二三四五六七八九十百千万零〇]+)", title)
        alpha = re.match(r"^\s*([A-Za-z]+)(\d+(?:\s*[.．]\s*\d+)*)", title)
        if appendix:
            return (2, (str(appendix.group(1)).casefold(),), item.get("page", 0))
        if alpha:
            return (
                2,
                (str(alpha.group(1)).casefold(), *[int(value) for value in re.findall(r"\d+", alpha.group(2))]),
                item.get("page", 0),
            )
        return (3, (), item.get("page", 0))

    outline.sort(key=sort_key)
    return outline


def extract_text_outline(chunks, full_text: str = "") -> list:
    """Fallback outline for text/PDF previews without layout cache."""
    outline = []
    seen = set()
    sources = []
    if isinstance(chunks, list):
        sources.extend(
            ((c.get("page_num") or 1) if isinstance(c, dict) else 1,
             c.get("text", "") if isinstance(c, dict) else "")
            for c in chunks
        )
    if not sources and full_text:
        sources = [(1, full_text)]
    for page_num, text in sources:
        for line in str(text or "").splitlines():
            stripped = line.strip()
            is_markdown_heading = bool(re.match(r"^#{1,6}\s+", stripped))
            if not is_markdown_heading and (
                len(stripped) > 80 or re.search(r"[。；;，,:：]$", stripped)
            ):
                continue
            parsed = _outline_title_level(line, "text", allow_single_number=True)
            if not parsed:
                continue
            title, level = parsed
            key = re.sub(r"\s+", " ", title).strip().casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            outline.append({
                "id": f"text-outline-{len(outline)}",
                "anchor": "",
                "title": title,
                "level": level,
                "page": int(page_num or 1),
            })
    return outline


def render_layout_html(blocks, page_num: int = 1, editable: bool = False) -> str:
    """Render cached PaddleOCR layout blocks using the OCR compare renderer.

    The OCR pipeline returns the actual block coordinates and content.  Keep
    this renderer in the ingestion module so the persisted-layout preview and
    the live OCR comparison cannot drift apart.
    """
    if not blocks:
        return ""

    items = []
    layout_meta = []
    records = []
    min_x = None
    max_x = max_y = 0
    for source_index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        box = (
            block.get("bbox")
            or block.get("block_bbox")
            or block.get("block_box")
            or block.get("box")
        )
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue

        raw_content = _clean_fragmentary_html(str(block.get("block_content", "")).strip())
        content = _normalize_ocr_escaped_newlines(raw_content)
        kind = str(
            block.get("block_type")
            or block.get("block_label")
            or block.get("type")
            or "text"
        ).lower()
        visual_data_uri = str(block.get("visual_data_uri") or "").strip()
        is_visual = _is_visual_block_kind(kind)
        if not content and not (is_visual and visual_data_uri.startswith("data:image/")):
            continue
        min_x = x1 if min_x is None else min(min_x, x1)
        max_x, max_y = max(max_x, x2), max(max_y, y2)
        block_index = len(records)
        records.append({
            "index": block_index,
            "source_index": source_index,
            "block": block,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "kind": kind,
            "content": content,
            "is_visual": is_visual,
            "visual_data_uri": visual_data_uri,
        })

    if not records:
        return ""

    # Pipeline may return small text/formula blocks inside a chart bbox.  The
    # chart crop already contains those pixels, so hide only the visual HTML
    # duplicate while keeping the text in OCR/chunks for search and copying.
    for visual in (
        record for record in records
        if record["is_visual"] and record["visual_data_uri"].startswith("data:image/")
    ):
        visual_area = max(1.0, (visual["x2"] - visual["x1"]) * (visual["y2"] - visual["y1"]))
        for candidate in records:
            if candidate is visual or candidate["is_visual"] or "title" in candidate["kind"]:
                continue
            if candidate["kind"] not in {
                "text", "paragraph", "caption", "inline_formula",
                "display_formula", "formula", "equation",
            }:
                continue
            intersection_width = max(
                0.0, min(visual["x2"], candidate["x2"]) - max(visual["x1"], candidate["x1"])
            )
            intersection_height = max(
                0.0, min(visual["y2"], candidate["y2"]) - max(visual["y1"], candidate["y1"])
            )
            candidate_area = max(
                1.0, (candidate["x2"] - candidate["x1"]) * (candidate["y2"] - candidate["y1"])
            )
            coverage = intersection_width * intersection_height / candidate_area
            center_x = (candidate["x1"] + candidate["x2"]) / 2
            center_y = (candidate["y1"] + candidate["y2"]) / 2
            if coverage >= 0.80 or (
                coverage >= 0.55
                and visual["x1"] <= center_x <= visual["x2"]
                and visual["y1"] <= center_y <= visual["y2"]
            ):
                candidate["covered_by_visual"] = True

    # The OCR box is a detection box, not always the intended paragraph
    # measure.  Use the page's right text boundary for ordinary flow text so
    # short lines do not wrap early just because the detector box was narrow.
    page_width = max_x + max(0, min_x or 0)
    # Layout coordinates are in source-image pixels.  A fixed browser font
    # becomes tiny when a 150/200-DPI page is fitted into the preview width.
    # Scale the source font with the page coordinate width so the fitted page
    # keeps a document-like reading size without changing its geometry.
    page_font_size = min(48.0, max(14.0, page_width * 0.015))
    # Do not derive text margins from every OCR block: headers, page numbers,
    # stamps and visual assets can start at x=0, which previously reduced the
    # right margin to zero while ordinary paragraphs still began inset from
    # the left edge.  Flow text establishes the readable column instead.
    flow_kinds = {"text", "paragraph", "caption", "footnote", "vision_footnote"}
    flow_lefts = [
        record["x1"] for record in records
        if record["kind"] in flow_kinds and not record["is_visual"]
    ]
    right_margin = max(0, min(flow_lefts) if flow_lefts else (min_x or 0))
    for record in records:
        x1 = record["x1"]
        y1 = record["y1"]
        x2 = record["x2"]
        y2 = record["y2"]
        kind = record["kind"]
        content = record["content"]
        is_visual = record["is_visual"] and bool(record["visual_data_uri"])
        is_table = "table" in kind or "<table" in content.lower()
        is_toc = (
            not is_table
            and kind in {"content", "contents", "toc", "table_of_contents", "tableofcontents"}
        ) or (not is_table and _looks_like_toc(content))
        if is_visual:
            rendered = (
                f'<img class="layout-visual-asset" '
                f'src="{html_lib.escape(record["visual_data_uri"], quote=True)}" '
                f'alt="">'
            )
        elif is_toc:
            rendered = _render_toc_html(content)
        elif is_table:
            rendered = _sanitize_layout_content(content)
            if editable:
                rendered = _decorate_editable_table_images(rendered)
        elif "formula" in kind or "equation" in kind:
            rendered = html_lib.escape(content)
        else:
            rendered = html_lib.escape(_compact_layout_text(content))
        if record.get("covered_by_visual"):
            rendered = ""
        is_heading = "title" in kind or "heading" in kind
        is_main_title = kind in {
            "title",
            "doc_title",
            "document_title",
            "main_title",
            "header_title",
        }
        is_flow_text = (
            not is_table
            and "formula" not in kind
            and "equation" not in kind
            and not is_heading
            and not is_toc
            and not is_visual
            and kind in {"text", "paragraph", "caption", "footnote", "vision_footnote"}
        )
        block_class = html_lib.escape(kind, quote=True)
        if is_heading:
            block_class += " ocr-heading ocr-centered-heading"
        if is_main_title:
            block_class += " ocr-main-title"
        if is_visual:
            block_class += " ocr-visual-block"
        if record.get("covered_by_visual"):
            block_class += " ocr-covered-by-visual"
        is_formula_like = "formula" in kind or "equation" in kind
        if is_formula_like:
            block_class += " formula-like"
            if "number" in kind:
                block_class += " formula-number-like"
        render_x1 = 0 if is_heading else x1
        render_x2 = page_width if is_heading else (max_x if is_flow_text else x2)
        block_height = max(1, y2 - y1)
        if is_flow_text:
            box_style = (
                f'left:{x1}px;top:{y1}px;right:{right_margin}px;'
                'min-height:1px;'
            )
        else:
            box_style = (
                f'left:{x1}px;top:{y1}px;width:{x2-x1}px;'
                f'min-height:{y2-y1}px;'
            )
        if is_visual:
            box_style += 'padding:0;overflow:hidden;'
        if is_formula_like:
            # MathJax may replace a short OCR box with a taller display node.
            # Keep a stable row box so its content can be centered, then let
            # the post-typeset alignment below move its matching number.
            box_style += f'height:{block_height}px'
        block_index = record["index"]
        anchor_id = ""
        if _outline_candidate(record["block"]):
            anchor_id = _outline_anchor_id(page_num, record["block"], record["source_index"])
        anchor_attr = (
            f' id="{anchor_id}" data-outline-id="{anchor_id}"'
            if anchor_id else ""
        )
        content_attrs = ""
        delete_control = ""
        if (
            editable
            and not is_visual
            and not is_formula_like
            and not record.get("covered_by_visual")
        ):
            content_format = "html" if is_table else "text"
            content_attrs = (
                ' contenteditable="true" spellcheck="false" '
                'data-layout-edit-content="true" '
                f'data-content-format="{content_format}" '
                f'data-original-content="{html_lib.escape(content, quote=True)}"'
            )
        elif editable and is_visual:
            delete_control = (
                '<button type="button" class="layout-delete-visual" '
                'title="删除图片">删除图片</button>'
            )
        elif editable and is_formula_like:
            delete_control = (
                '<button type="button" class="layout-edit-formula" '
                f'data-original-content="{html_lib.escape(content, quote=True)}" '
                f'data-current-content="{html_lib.escape(content, quote=True)}" '
                'title="编辑公式">编辑公式</button>'
            )
        items.append(
            f'<div class="ocr-block ocr-{block_class}"{anchor_attr} '
            f'data-layout-id="layout-{block_index}" '
            f'data-source-index="{record["source_index"]}" '
            f'data-x1="{render_x1}" data-x2="{render_x2}" data-top="{y1}" '
            f'style="{box_style}">'
            f'<div class="ocr-block-content"{content_attrs}>{rendered}</div>'
            f'{delete_control}</div>'
        )
        layout_meta.append({
            "index": block_index,
            "kind": kind,
            "x1": x1,
            "x2": render_x2,
            "y1": y1,
            "y2": y2,
        })

    # Formula numbers are separate OCR blocks.  Reconstruct the conventional
    # right-aligned equation-number leader when the two blocks share a row.
    for formula in layout_meta:
        if (
            not ("formula" in formula["kind"] or "equation" in formula["kind"])
            or "number" in formula["kind"]
        ):
            continue
        candidates = [
            number for number in layout_meta
            if "number" in number["kind"]
            and number["x2"] > formula["x2"]
            and min(formula["y2"], number["y2"]) > max(formula["y1"], number["y1"])
        ]
        if not candidates:
            continue
        number = min(candidates, key=lambda item: item["x1"])
        formula_id = f'layout-{formula["index"]}'
        number_id = f'layout-{number["index"]}'
        items[formula["index"]] = items[formula["index"]].replace(
            f'data-layout-id="{formula_id}"',
            f'data-layout-id="{formula_id}" data-formula-peer-id="{number_id}"',
            1,
        )
        items[number["index"]] = items[number["index"]].replace(
            f'data-layout-id="{number_id}"',
            f'data-layout-id="{number_id}" data-formula-peer-id="{formula_id}"',
            1,
        )
        line_start = max(formula["x2"], number["x1"])
        line_end = number["x2"]
        gap = line_end - line_start
        if gap < 12:
            continue
        center_y = (max(formula["y1"], number["y1"]) + min(formula["y2"], number["y2"])) / 2
        dash_count = max(3, min(32, int(gap / 6)))
        items.append(
            f'<div class="formula-connector" '
            f'data-formula-id="{formula_id}" '
            f'data-number-id="{number_id}" '
            f'style="left:{line_start}px;top:{center_y - 8}px;'
            f'width:{gap}px;height:16px">{"-" * dash_count}</div>'
        )

    edit_script = ""
    edit_style = ""
    if editable:
        edit_script = (
            '<script>(function(){'
            f'var pageNum={int(page_num or 1)};'
            'function send(payload){if(window.parent&&window.parent!==window)'
            'window.parent.postMessage(Object.assign({type:"rongneng-layout-edit",page_num:pageNum},payload),"*");}'
            'document.addEventListener("input",function(event){'
            'var target=event.target.closest("[data-layout-edit-content]");if(!target)return;'
            'var block=target.closest(".ocr-block");if(!block)return;'
            'var index=Number(block.dataset.sourceIndex),format=target.dataset.contentFormat||"text";'
            'send({op:"update",block_index:index,content:format==="html"?target.innerHTML:target.innerText,'
            'content_format:format,before:target.dataset.originalContent||""});'
            'window.requestAnimationFrame(function(){if(window.rongnengRefreshLayout)window.rongnengRefreshLayout();});'
            '});'
            'document.addEventListener("click",function(event){'
            'var button=event.target.closest(".layout-delete-visual");if(!button)return;'
            'event.preventDefault();event.stopPropagation();var block=button.closest(".ocr-block");if(!block)return;'
            'block.classList.add("is-edit-deleted");send({op:"delete",block_index:Number(block.dataset.sourceIndex),'
            'before:"[图片]",content:"",content_format:"text"});'
            '});'
            'document.addEventListener("click",function(event){'
            'var button=event.target.closest(".layout-delete-inline-image");if(!button)return;'
            'event.preventDefault();event.stopPropagation();var wrap=button.closest(".layout-inline-image"),'
            'block=button.closest(".ocr-block"),imageIndex=Number(button.dataset.inlineImageIndex);'
            'if(!wrap||!block||!Number.isInteger(imageIndex)||imageIndex<0)return;wrap.remove();send({op:"delete_table_image",'
            'block_index:Number(block.dataset.sourceIndex),image_index:imageIndex,before:"[表格内图片]",content:"",content_format:"html"});'
            '});'
            'document.addEventListener("click",function(event){'
            'var button=event.target.closest(".layout-edit-formula");if(!button)return;'
            'event.preventDefault();event.stopPropagation();var block=button.closest(".ocr-block");if(!block)return;'
            'var current=button.dataset.currentContent||"",next=window.prompt("编辑 LaTeX 公式",current);'
            'if(next===null||next===current)return;button.dataset.currentContent=next;'
            'var target=block.querySelector(".ocr-block-content");if(target)target.textContent=next;'
            'send({op:"update",block_index:Number(block.dataset.sourceIndex),content:next,content_format:"text",'
            'before:button.dataset.originalContent||""});'
            'if(target&&window.MathJax&&MathJax.typesetPromise){'
            'if(MathJax.typesetClear)MathJax.typesetClear([target]);'
            'MathJax.typesetPromise([target]).catch(function(){}).then(function(){'
            'if(window.rongnengRefreshLayout)window.rongnengRefreshLayout();});}'
            '});'
            'window.addEventListener("message",function(event){var data=event.data||{};'
            'if(data.type!=="rongneng-layout-edit-restore")return;'
            'var block=document.querySelector(".ocr-block[data-source-index=\\\""+Number(data.block_index)+"\\\"]");'
            'if(!block)return;if(data.op==="delete"){block.classList.remove("is-edit-deleted");return;}'
            'var target=block.querySelector("[data-layout-edit-content]");if(!target)return;'
            'if(data.content_format==="html")target.innerHTML=String(data.content||"");'
            'else target.innerText=String(data.content||"");'
            'var formulaButton=block.querySelector(".layout-edit-formula");'
            'if(formulaButton)formulaButton.dataset.currentContent=String(data.content||"");'
            'if(formulaButton&&window.MathJax&&MathJax.typesetPromise){'
            'if(MathJax.typesetClear)MathJax.typesetClear([target]);'
            'MathJax.typesetPromise([target]).catch(function(){}).then(function(){'
            'if(window.rongnengRefreshLayout)window.rongnengRefreshLayout();});}'
            '});'
            '})();</script>'
        )
        edit_style = (
            '.page.is-editable .ocr-block:not(.ocr-covered-by-visual){transition:box-shadow .12s ease,background .12s ease;}'
            '.page.is-editable .ocr-block:not(.ocr-visual-block):not(.ocr-covered-by-visual):hover{'
            'box-shadow:0 0 0 2px rgba(47,111,219,.34);background:rgba(233,241,255,.28);z-index:3}'
            '[data-layout-edit-content]{outline:none;cursor:text;border-radius:2px}'
            '[data-layout-edit-content]:focus{box-shadow:0 0 0 2px #2f6fdb;background:rgba(255,255,255,.96)}'
            '.page.is-editable .ocr-visual-block{overflow:visible!important;z-index:2}'
            '.page.is-editable .ocr-visual-block>.ocr-block-content{overflow:hidden}'
            '.layout-delete-visual{position:absolute;right:8px;top:8px;display:none;border:0;border-radius:3px;'
            'padding:7px 10px;background:#b42318;color:#fff;font:600 13px/1 Arial,"Microsoft YaHei",sans-serif;'
            'cursor:pointer;box-shadow:0 2px 8px rgba(0,0,0,.22)}'
            '.ocr-visual-block:hover>.layout-delete-visual{display:block}'
            '.layout-inline-image{position:relative;display:inline-block;max-width:100%;line-height:0}'
            '.layout-inline-image>img{display:block;max-width:100%;height:auto}'
            '.layout-delete-inline-image{position:absolute;right:3px;top:3px;display:none;border:0;border-radius:3px;'
            'padding:3px 5px;background:#d84a4a;color:#fff;font-size:11px;line-height:1.2;cursor:pointer}'
            '.layout-inline-image:hover>.layout-delete-inline-image{display:block}'
            '.layout-edit-formula{position:absolute;right:2px;top:-30px;display:none;border:1px solid #2f6fdb;'
            'border-radius:3px;padding:5px 8px;background:#fff;color:#1f5fbf;font:600 12px/1 Arial,"Microsoft YaHei",sans-serif;'
            'cursor:pointer;white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,.12)}'
            '.formula-like:hover>.layout-edit-formula{display:block}'
            '.ocr-block.is-edit-deleted{display:none!important}'
        )

    page_edit_class = " is-editable" if editable else ""
    return (
        '<!doctype html><meta charset="utf-8">'
        '<script>window.MathJax={tex:{inlineMath:[["$","$"],["\\\\(","\\\\)"]],'
        'displayMath:[["$$","$$"],["\\\\[","\\\\]"]]},startup:{typeset:false}};'
        '(function(){'
        'function overlapX(a,b){return Math.min(a.x2,b.x2)-Math.max(a.x1,b.x1)>2;}'
        'function settle(){'
        'var page=document.querySelector(".page");'
        'if(!page)return;'
        'var nodes=Array.from(document.querySelectorAll(".ocr-block:not(.ocr-visual-block):not(.ocr-covered-by-visual)"));'
        'nodes.sort(function(a,b){return parseFloat(a.dataset.top)-parseFloat(b.dataset.top)||parseFloat(a.dataset.x1)-parseFloat(b.dataset.x1);});'
        'var placed=[],bottom=parseFloat(page.dataset.baseHeight||"0");'
        'nodes.forEach(function(node){'
        'var item={node:node,x1:parseFloat(node.dataset.x1),x2:parseFloat(node.dataset.x2),top:parseFloat(node.dataset.top),height:0};'
        'placed.forEach(function(prev){if(overlapX(item,prev)&&item.top<prev.top+prev.height)item.top=prev.top+prev.height+2;});'
        'node.style.top=item.top+"px";'
        'item.height=node.getBoundingClientRect().height;'
        'placed.push(item);'
        'bottom=Math.max(bottom,item.top+item.height);'
        '});'
        'page.style.minHeight=(bottom+12)+"px";'
        '}'
        'function alignFormulaRows(){'
        'var page=document.querySelector(".page");'
        'if(!page)return;'
        'Array.from(page.querySelectorAll(".formula-like:not(.formula-number-like)[data-formula-peer-id]")).forEach(function(formula){'
        'var number=page.querySelector("[data-layout-id=\\""+formula.dataset.formulaPeerId+"\\"]");'
        'if(!formula||!number)return;'
        'var formulaRect=formula.getBoundingClientRect();'
        'var numberRect=number.getBoundingClientRect();'
        'var formulaCenter=formulaRect.top+formulaRect.height/2;'
        'var numberCenter=numberRect.top+numberRect.height/2;'
        'var delta=formulaCenter-numberCenter;'
        'if(Math.abs(delta)>0.5){'
        'var current=parseFloat(number.style.top||"0");'
        'number.style.top=(current+delta)+"px";'
        '}'
        'var left=formula.offsetLeft+formula.offsetWidth;'
        'var right=number.offsetLeft;'
        'var center=formula.offsetTop+formula.offsetHeight/2;'
        'var line=page.querySelector(".formula-connector[data-formula-id=\\""+formula.dataset.layoutId+"\\"]");'
        'if(line){'
        'line.style.left=left+"px";'
        'line.style.width=Math.max(0,right-left)+"px";'
        'line.style.top=(center-line.offsetHeight/2)+"px";'
        '}'
        '});'
        '}'
        'function fitPage(){'
        'var page=document.querySelector(".page");'
        'if(!page)return;'
        'var naturalWidth=page.offsetWidth;'
        'var availableWidth=Math.max(320,window.innerWidth-24);'
        'var scale=Math.min(1,availableWidth/naturalWidth);'
        'page.style.transformOrigin="top left";'
        'page.style.transform="scale("+scale+")";'
        'var scaledWidth=naturalWidth*scale;'
        'var scaledHeight=page.offsetHeight*scale;'
        'page.style.marginLeft=Math.max(8,(window.innerWidth-scaledWidth)/2)+"px";'
        'page.style.marginRight=Math.max(8,(window.innerWidth-scaledWidth)/2)+"px";'
        'document.documentElement.style.overflow="hidden";'
        'document.body.style.overflow="hidden";'
        'document.body.style.height=(scaledHeight+16)+"px";'
        'if(window.parent&&window.parent!==window){'
        'window.parent.postMessage({type:"rongneng-layout-size",height:Math.ceil(scaledHeight+16)},"*");'
        '}'
        '}'
        'window.rongnengRefreshLayout=function(){settle();alignFormulaRows();fitPage();};'
        'function reveal(){'
        'var page=document.querySelector(".page");'
        'if(!page)return;'
        'var fonts=(document.fonts&&document.fonts.ready)||Promise.resolve();'
        'Promise.resolve(fonts).catch(function(){}).then(function(){'
        'requestAnimationFrame(function(){requestAnimationFrame(function(){settle();alignFormulaRows();fitPage();page.style.visibility="visible";});});'
        '});'
        '}'
        'window.addEventListener("load",function(){'
        'var run=function(){'
        'if(window.MathJax&&MathJax.typesetPromise){'
        'var startup=(MathJax.startup&&MathJax.startup.promise)||Promise.resolve();'
        'startup.then(function(){return MathJax.typesetPromise([document.body]);}).catch(function(){}).then(reveal);'
        '}else{reveal();}'
        '};'
        'run();'
        '});'
        'window.addEventListener("resize",fitPage);'
        'function reportAnchor(anchor){'
        'var target=document.getElementById(anchor),page=document.querySelector(".page");'
        'if(!target||!page||!window.parent||window.parent===window)return;'
        'var targetRect=target.getBoundingClientRect(),pageRect=page.getBoundingClientRect();'
        'window.parent.postMessage({type:"rongneng-layout-anchor",anchor:anchor,offset:Math.max(0,targetRect.top-pageRect.top)},"*");'
        '}'
        'window.addEventListener("message",function(event){'
        'var data=event.data||{};'
        'if(data.type==="rongneng-layout-scroll"&&data.anchor)reportAnchor(String(data.anchor));'
        '});'
        '})();</script>'
        + edit_script +
        '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'
        '<style>'
        'html,body{margin:0;background:#f3f4f6;overflow:hidden} .page{position:relative;'
        f'width:{page_width}px;min-height:{max_y}px;margin:8px auto;background:#fff;visibility:hidden;'
        f'box-shadow:0 1px 6px #bbb;font:{page_font_size:.1f}px/1.45 Arial,"Microsoft YaHei",sans-serif}}'
        '.ocr-block{position:absolute;overflow:visible;box-sizing:border-box;white-space:pre-wrap;'
        'word-break:break-word;overflow-wrap:anywhere;padding:1px}'
        '.ocr-block-content{width:100%;max-width:100%;min-height:100%;box-sizing:border-box;'
        'overflow:visible;overflow-wrap:anywhere}'
        '.ocr-visual-block{padding:0!important;overflow:hidden!important;z-index:0}'
        '.ocr-visual-block>.ocr-block-content{width:100%;height:100%;min-height:0}'
        '.layout-visual-asset{display:block;width:100%;height:100%;object-fit:fill;}'
        '.ocr-covered-by-visual{display:none!important}'
        '.layout-toc{display:flex;flex-direction:column;gap:2px;width:100%;'
        'font-size:.92em;line-height:1.45}'
        '.layout-toc-row{display:flex;align-items:baseline;min-height:1.35em;'
        'white-space:nowrap;gap:6px;min-width:0}'
        '.layout-toc-title{min-width:0;overflow:hidden;text-overflow:ellipsis}'
        '.layout-toc-leader{flex:1 1 auto;min-width:12px;height:.7em;'
        'border-bottom:1px dotted #8c8c8c}'
        '.layout-toc-page{flex:0 0 auto;min-width:2.5em;text-align:right;'
        'font-variant-numeric:tabular-nums;color:#444}'
        '.formula-like{display:flex;align-items:center}'
        '.formula-like>.ocr-block-content{height:100%;display:flex;align-items:center;justify-content:center}'
        '.formula-number-like>.ocr-block-content{justify-content:flex-end}'
        '.formula-like mjx-container[display="true"]{margin:0!important}'
        '.ocr-block table{border-collapse:collapse;width:100%;height:auto;table-layout:fixed}'
        '.ocr-block td,.ocr-block th{border:1px solid #666;padding:3px;text-align:center;'
        'vertical-align:middle;overflow:visible;word-break:break-word;overflow-wrap:anywhere;'
        'white-space:pre-line}'
        '.ocr-heading{white-space:nowrap;word-break:keep-all;width:max-content!important;}'
        '.ocr-centered-heading{left:0!important;right:0!important;width:auto!important;'
        'text-align:center;white-space:nowrap;word-break:keep-all;font-weight:600;}'
        '.ocr-main-title{font-size:1.35em;font-weight:700;}'
        '.ocr-block.ocr-number{white-space:nowrap;word-break:normal;overflow:visible;'
        'width:max-content!important;min-width:20px}'
        '.ocr-block.ocr-formula_number{white-space:nowrap;word-break:normal;'
        'overflow:visible;text-align:right;z-index:1}'
        '.formula-connector{position:absolute;z-index:0;overflow:hidden;white-space:nowrap;'
        'text-align:center;line-height:16px;color:#555;letter-spacing:1px;pointer-events:none}'
        + edit_style +
        f'</style><div class="page{page_edit_class}">' + "".join(items) + "</div>"
    )


def render_layout_pages_html(layout_pages, editable: bool = False) -> list:
    """Render one PNG-like HTML document per PDF page.

    PDF layout cache keys are zero-based page indexes because they come from
    ``recognize_pdf_pages``.  Keep the cache convention internal and expose
    one-based page numbers to the API/UI.  A plain block list is treated as a
    single-page image, which keeps PNG behavior unchanged.
    """
    if isinstance(layout_pages, dict):
        entries = []
        for raw_page, blocks in layout_pages.items():
            try:
                page_num = int(raw_page) + 1
            except (TypeError, ValueError):
                continue
            entries.append((page_num, blocks if isinstance(blocks, list) else []))
        entries.sort(key=lambda item: item[0])
    elif isinstance(layout_pages, list):
        entries = [(1, layout_pages)]
    else:
        entries = []

    return [
        {
            "page_num": page_num,
            "layout_html": render_layout_html(
                blocks, page_num=page_num, editable=editable
            ),
        }
        for page_num, blocks in entries
    ]


def _compile_repeat_re():
    """编译用于检测重复表格单元的表达式 (惰性)"""
    global _REPEAT_CELL_RE
    if _REPEAT_CELL_RE is None:
        # 匹配 "词元 | 数值 | " 重复 >= 8 次 或 "词元 | 数值 | 词元 | 数值" 循环
        _REPEAT_CELL_RE = __import__("re").compile(
            r"((?:[一-鿿A-Za-z0-9]+[ \t]*\|[ \t]*\d+[ \t]*\|[ \t]*){8,})"
        )
    return _REPEAT_CELL_RE


def is_garbage_ocr_text(text: str) -> bool:
    """
    检测 OCR 结果是否为小模型幻觉/垃圾输出。

    垃圾特征:
      1. 模型自我解释废话 (含 _ILLUSION_PHRASES)
      2. 高度重复的表格单元循环 (如 "名称|100|名称|100..." 重复)
      3. 极短且无实际信息量 (纯符号/噪声)

    Returns:
        True 表示垃圾, 应丢弃该页
    """
    if not text or not text.strip():
        return True

    t = text.strip()

    # 特征1: 幻觉废话
    for phrase in _ILLUSION_PHRASES:
        if phrase in t:
            return True

    # 特征2: 表格单元重复循环 (如 "名称|100" × N)
    _re = _compile_repeat_re()
    if _re.search(t):
        # 确认是长重复 (重复部分占文本大部分)
        m = _re.search(t)
        if m and len(m.group(1)) >= len(t) * 0.5:
            return True

    # 特征3: 单一词元疯狂重复 (如 "名称名称名称...")
    import re as _re2
    toks = _re2.findall(r"[一-鿿]{2,4}", t)
    if toks:
        most_common = max(set(toks), key=toks.count)
        if toks.count(most_common) >= 10 and toks.count(most_common) >= len(toks) * 0.5:
            return True

    return False


class VLOcrClient:
    """调用外部 PaddleOCR-VL 服务 (OpenAI 兼容协议) 进行结构化 OCR"""

    def __init__(self, base_url: str, model: str = "paddleocr-vl-1.6",
                 timeout: int = 180, dpi: int = 150, max_image_dim: int = 3000,
                 max_tokens: int = 1024, protocol: str = "openai",
                 endpoint: str = "/ocr"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.dpi = dpi
        self.max_image_dim = max_image_dim
        self.max_tokens = max_tokens
        self.protocol = (protocol or "openai").lower()
        self.endpoint = endpoint or "/ocr"
        self.last_layout_blocks = []
        self.last_layout_pages = {}
        self._client = None
        self._ready = False
        if self.protocol not in {"pipeline", "paddleocr_pipeline", "multipart"}:
            self._ensure_client()

    def _ensure_client(self):
        """延迟初始化 OpenAI 客户端 (线程安全由调用方保证)"""
        if self._ready:
            return
        from openai import OpenAI
        self._client = OpenAI(
            base_url=f"{self.base_url}/v1",
            api_key="sk-empty",  # 本地服务通常不校验 key
            timeout=self.timeout,
        )
        self._ready = True
        print(f"[vl-ocr] PaddleOCR-VL 服务: {self.base_url}, model={self.model}")

    def _render_page_to_png(self, page: fitz.Page) -> bytes:
        """将 PDF 页面渲染为 PNG bytes"""
        pix = page.get_pixmap(dpi=self.dpi)

        # 大图缩放 (限制最大边长, 控制请求体积)
        if self.max_image_dim > 0 and max(pix.width, pix.height) > self.max_image_dim:
            from PIL import Image
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            img.thumbnail((self.max_image_dim, self.max_image_dim), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        return pix.tobytes("png")

    def enrich_layout_pages_with_visual_assets(self, pdf_path: str, layout_pages: dict) -> dict:
        """Backfill chart/image crops into an existing PDF layout cache.

        This path does not call OCR and does not touch chunks or vectors.  It
        uses the same DPI and max-image scaling as ``recognize_pdf_pages`` so a
        previously cached bbox remains aligned with the generated crop.
        """
        if not isinstance(layout_pages, dict):
            return layout_pages
        enriched = {}
        doc = fitz.open(pdf_path)
        try:
            for raw_page, blocks in layout_pages.items():
                try:
                    page_index = int(raw_page)
                except (TypeError, ValueError):
                    enriched[raw_page] = blocks
                    continue
                if not isinstance(blocks, list) or page_index < 0 or page_index >= len(doc):
                    enriched[raw_page] = blocks
                    continue
                image = self._render_page_to_png(doc[page_index])
                enriched[raw_page] = _attach_visual_assets(image, blocks)
        finally:
            doc.close()
        return enriched

    def recognize_image(self, image_bytes: bytes) -> str:
        """识别单张图片, 返回结构化文本 (markdown)"""
        if self.protocol in {"pipeline", "paddleocr_pipeline", "multipart"}:
            text = self._recognize_pipeline(image_bytes)
            self.last_layout_blocks = _attach_visual_assets(
                image_bytes, self.last_layout_blocks
            )
            return text
        self._ensure_client()
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _VL_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            temperature=0,
            max_tokens=self.max_tokens,
        )
        content = resp.choices[0].message.content
        return (content or "").strip()

    def _recognize_pipeline(self, image_bytes: bytes) -> str:
        """Call PaddleOCR pipeline's multipart ``POST /ocr`` endpoint.

        The pipeline returns layout blocks rather than an OpenAI chat response.
        Keep only ordered textual blocks so downstream chunking/indexing sees the
        same plain/markdown text contract as the OpenAI-compatible client.
        """
        import httpx

        url = f"{self.base_url}{self.endpoint if self.endpoint.startswith('/') else '/' + self.endpoint}"
        print(f"[OCR-TRACE] POST pipeline {url}", flush=True)
        response = httpx.post(
            url,
            files={"file": ("page.png", image_bytes, "image/png")},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("result", []) if isinstance(payload, dict) else []
        if isinstance(results, dict):
            results = [results]
        blocks = []
        self.last_layout_blocks = []
        for item in results:
            res = item.get("res", item) if isinstance(item, dict) else {}
            parsing = res.get("parsing_res_list", [])
            if isinstance(parsing, list) and parsing:
                self.last_layout_blocks.extend(
                    block.copy() for block in parsing if isinstance(block, dict)
                )
                def sort_key(block):
                    xy = _block_xy(block)
                    # 坐标优先于 block_order：复杂表格页上后者可能把页尾
                    # 段落（例如 3.0.4）错误排到表格前。
                    if xy is not None:
                        return (0, xy[0], xy[1], _number(block.get("block_order")), _number(block.get("block_id")))
                    return (1, _number(block.get("block_order")), _number(block.get("block_id")), 0, 0)

                ordered = sorted(
                    (block for block in parsing if isinstance(block, dict)),
                    key=sort_key,
                )
                blocks.extend(_dedupe_blocks(
                    _normalize_ocr_escaped_newlines(
                        str(block.get("block_content", "")).strip()
                    )
                    for block in ordered
                    if str(block.get("block_content", "")).strip()
                ))
            else:
                markdown = res.get("markdown") or res.get("markdown_text")
                if isinstance(markdown, str) and markdown.strip():
                    blocks.append(
                        _dedupe_markdown(_normalize_ocr_escaped_newlines(markdown))
                    )
        # Keep image data out of the text returned for chunking/indexing. Only
        # the persisted layout blocks need self-contained table image assets.
        self.last_layout_blocks = _inline_pipeline_table_images(
            image_bytes, self.last_layout_blocks
        )
        return "\n\n".join(_dedupe_blocks(blocks)).strip()

    def recognize_pdf_pages(self, pdf_path: str,
                            pages_0based: List[int]) -> Dict[int, str]:
        """
        对 PDF 指定页面执行结构化 OCR

        Args:
            pdf_path: PDF 文件路径
            pages_0based: 需要识别的页面索引 (0-based)

        Returns:
            {0-based_page_index: 结构化文本}
        """
        result: Dict[int, str] = {}
        self.last_layout_pages = {}
        if not pages_0based:
            return result

        doc = fitz.open(pdf_path)
        try:
            for page_idx in pages_0based:
                if page_idx >= len(doc):
                    continue
                t0 = time.time()
                try:
                    img = self._render_page_to_png(doc[page_idx])
                    text = self.recognize_image(img)
                    self.last_layout_pages[page_idx] = list(self.last_layout_blocks)
                    if text:
                        result[page_idx] = text
                    else:
                        print(f"   [vl-ocr] 第{page_idx+1}页无有效内容")
                except Exception as e:
                    print(f"   [vl-ocr] 第{page_idx+1}页识别失败: {e}")
                elapsed = time.time() - t0
                print(f"   [vl-ocr] 第{page_idx+1}页 → {len(result.get(page_idx,''))}字符 "
                      f"({elapsed:.1f}s)")
        finally:
            doc.close()
        return result
