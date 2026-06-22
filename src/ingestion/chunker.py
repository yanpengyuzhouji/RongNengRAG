"""
格式感知分块引擎 — 根据文档类型选择分块策略
三档分块：
  1. 语义分块：多页标准规范/通知文件
  2. 单页即块：CAD 导出 PDF、短通知
  3. 文档即块：极短文件不拆分
"""

import re
import sqlite3
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """单个文本块"""
    chunk_id: str                    # 唯一标识: {file_hash}_{chunk_index}
    file_hash: str
    text: str
    char_count: int

    # 元数据（从文件信息继承）
    domain: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    file_path: str = ""
    file_name: str = ""
    file_type: str = ""
    page_num: Optional[int] = None
    total_pages: int = 0
    chunk_index: int = 0
    total_chunks: int = 0

    # 标签元数据
    doc_number: Optional[str] = None
    publish_level: Optional[str] = None
    voltage_level: Optional[str] = None
    discipline: Optional[str] = None
    equipment_type: Optional[str] = None
    year: Optional[int] = None
    region: str = "全国"
    drawing_code: Optional[str] = None
    is_drawing: bool = False
    is_archive: bool = False
    format_group: str = "document"

    # 分块元数据
    chunk_strategy: str = "semantic"  # "semantic" | "single_page" | "full_document"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "file_hash": self.file_hash,
            "text": self.text,
            "char_count": self.char_count,
            "domain": self.domain,
            "category": self.category,
            "subcategory": self.subcategory,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "page_num": self.page_num,
            "total_pages": self.total_pages,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "doc_number": self.doc_number,
            "publish_level": self.publish_level,
            "voltage_level": self.voltage_level,
            "discipline": self.discipline,
            "equipment_type": self.equipment_type,
            "year": self.year,
            "region": self.region,
            "drawing_code": self.drawing_code,
            "is_drawing": self.is_drawing,
            "is_archive": self.is_archive,
            "format_group": self.format_group,
            "chunk_strategy": self.chunk_strategy,
        }

    def get_metadata_str(self) -> str:
        """生成用于检索增强的元数据文本"""
        parts = []
        if self.domain:
            parts.append(f"专业域:{self.domain}")
        if self.category:
            parts.append(f"文档类目:{self.category}")
        if self.doc_number:
            parts.append(f"文件编号:{self.doc_number}")
        if self.publish_level:
            parts.append(f"发布层级:{self.publish_level}")
        if self.voltage_level:
            parts.append(f"电压等级:{self.voltage_level}")
        if self.discipline:
            parts.append(f"专业类型:{self.discipline}")
        if self.equipment_type:
            parts.append(f"设备类型:{self.equipment_type}")
        if self.region:
            parts.append(f"地域:{self.region}")
        if self.year:
            parts.append(f"年份:{self.year}")
        if self.drawing_code:
            parts.append(f"图号:{self.drawing_code}")
        return " | ".join(parts)


class Chunker:
    """格式感知的分块器"""

    def __init__(self, config_path: str = None):
        from config import load_config
        self.config = load_config(config_path)

        chunk_config = self.config["chunking"]
        self.semantic_chunk_size = chunk_config["semantic"]["chunk_size"]
        self.semantic_chunk_overlap = chunk_config["semantic"]["chunk_overlap"]
        self.single_page_enabled = chunk_config["single_page"]["enabled"]
        self.max_pages_for_single = chunk_config["single_page"]["max_pages_for_single"]
        self.max_chars_for_full = chunk_config["full_document"]["max_chars_for_full"]

        # 中文优先分隔符
        self.separators = [
            "\n\n\n", "\n\n", "\n",
            "。", "；", "，",
            ". ", " ", ""
        ]

    def chunk_pdf_document(self, parsed_pdf: dict, file_meta: dict) -> List[Chunk]:
        """
        对解析后的 PDF 内容进行分块
        根据页数和文本量选择策略
        """
        page_count = parsed_pdf["page_count"]
        total_chars = parsed_pdf["total_chars"]
        all_text = "\n".join([p["text"] for p in parsed_pdf["pages"]])

        # 策略选择
        if total_chars < self.max_chars_for_full:
            # 策略3: 整个文档不拆分
            strategy = "full_document"
            chunks = [self._make_full_document_chunk(all_text, file_meta, page_count)]
        elif page_count <= self.max_pages_for_single and self.single_page_enabled:
            # 策略2: 单页即块
            strategy = "single_page"
            chunks = self._make_per_page_chunks(parsed_pdf["pages"], file_meta)
        else:
            # 策略1: 语义分块
            strategy = "semantic"
            chunks = self._make_semantic_chunks(all_text, file_meta, page_count)

        return chunks

    def chunk_text_document(self, text: str, file_meta: dict) -> List[Chunk]:
        """
        对纯文本文档（DOC/DOCX/OFD/CEB转换后）分块
        """
        if len(text) < self.max_chars_for_full:
            return [self._make_full_document_chunk(text, file_meta, 1)]

        return self._make_semantic_chunks(text, file_meta, 1)

    def chunk_drawing(self, text: str, file_meta: dict) -> List[Chunk]:
        """
        对图纸文件分块 — 每个文件一个 chunk，检索主要靠元数据
        """
        # 即使文本很少或为空，也创建一个 chunk
        # 检索信号来自元数据而非正文
        chunk = self._make_full_document_chunk(
            text if text else f"[CAD图纸] {file_meta.get('drawing_code', '')} {file_meta.get('file_name', '')}",
            file_meta, 1
        )
        chunk.chunk_strategy = "single_page"
        return [chunk]

    def _make_full_document_chunk(self, text: str, file_meta: dict, page_count: int) -> Chunk:
        """创建全文档 chunk"""
        return Chunk(
            chunk_id=f"{file_meta['file_hash']}_0",
            file_hash=file_meta["file_hash"],
            text=text.strip(),
            char_count=len(text.strip()),
            domain=file_meta.get("domain"),
            category=file_meta.get("category"),
            subcategory=file_meta.get("subcategory"),
            file_path=file_meta.get("relative_path", ""),
            file_name=file_meta.get("file_name", ""),
            file_type=file_meta.get("extension", ""),
            page_num=None,
            total_pages=page_count,
            chunk_index=0,
            total_chunks=1,
            doc_number=file_meta.get("doc_number"),
            publish_level=file_meta.get("publish_level"),
            voltage_level=file_meta.get("voltage_level"),
            discipline=file_meta.get("discipline"),
            equipment_type=file_meta.get("equipment_type"),
            year=file_meta.get("year"),
            region=file_meta.get("region", "全国"),
            drawing_code=file_meta.get("drawing_code"),
            is_drawing=bool(file_meta.get("is_drawing", 0)),
            is_archive=bool(file_meta.get("is_archive", 0)),
            format_group=file_meta.get("format_group", "document"),
            chunk_strategy="full_document",
        )

    def _make_per_page_chunks(self, pages: list, file_meta: dict) -> List[Chunk]:
        """每页一个 chunk"""
        chunks = []
        total = len(pages)
        for page in pages:
            text = page["text"]
            chunk = Chunk(
                chunk_id=f"{file_meta['file_hash']}_{page['page_num']}",
                file_hash=file_meta["file_hash"],
                text=text if text else f"[第{page['page_num']}页] {file_meta.get('file_name', '')}",
                char_count=len(text),
                domain=file_meta.get("domain"),
                category=file_meta.get("category"),
                subcategory=file_meta.get("subcategory"),
                file_path=file_meta.get("relative_path", ""),
                file_name=file_meta.get("file_name", ""),
                file_type=file_meta.get("extension", ""),
                page_num=page["page_num"],
                total_pages=total,
                chunk_index=page["page_num"] - 1,
                total_chunks=total,
                doc_number=file_meta.get("doc_number"),
                publish_level=file_meta.get("publish_level"),
                voltage_level=file_meta.get("voltage_level"),
                discipline=file_meta.get("discipline"),
                equipment_type=file_meta.get("equipment_type"),
                year=file_meta.get("year"),
                region=file_meta.get("region", "全国"),
                drawing_code=file_meta.get("drawing_code"),
                is_drawing=bool(file_meta.get("is_drawing", 0)),
                is_archive=bool(file_meta.get("is_archive", 0)),
                format_group=file_meta.get("format_group", "document"),
                chunk_strategy="single_page",
            )
            chunks.append(chunk)
        return chunks

    def _make_semantic_chunks(self, text: str, file_meta: dict, page_count: int) -> List[Chunk]:
        """语义分块 — 使用中文优先分隔符递归拆分"""
        # 先按字符数粗略拆分（中文约 1.5 字符 ≈ 1 token）
        max_chars = self.semantic_chunk_size * 3  # ~512 tokens * 3 chars/token
        overlap_chars = self.semantic_chunk_overlap * 3

        splits = self._recursive_split(text, max_chars, overlap_chars)
        chunks = []
        total = len(splits)

        for i, split_text in enumerate(splits):
            if not split_text.strip():
                continue

            chunk = Chunk(
                chunk_id=f"{file_meta['file_hash']}_{i}",
                file_hash=file_meta["file_hash"],
                text=split_text.strip(),
                char_count=len(split_text.strip()),
                domain=file_meta.get("domain"),
                category=file_meta.get("category"),
                subcategory=file_meta.get("subcategory"),
                file_path=file_meta.get("relative_path", ""),
                file_name=file_meta.get("file_name", ""),
                file_type=file_meta.get("extension", ""),
                page_num=None,
                total_pages=page_count,
                chunk_index=i,
                total_chunks=total,
                doc_number=file_meta.get("doc_number"),
                publish_level=file_meta.get("publish_level"),
                voltage_level=file_meta.get("voltage_level"),
                discipline=file_meta.get("discipline"),
                equipment_type=file_meta.get("equipment_type"),
                year=file_meta.get("year"),
                region=file_meta.get("region", "全国"),
                drawing_code=file_meta.get("drawing_code"),
                is_drawing=bool(file_meta.get("is_drawing", 0)),
                is_archive=bool(file_meta.get("is_archive", 0)),
                format_group=file_meta.get("format_group", "document"),
                chunk_strategy="semantic",
            )
            chunks.append(chunk)

        return chunks

    def _recursive_split(self, text: str, max_chars: int, overlap: int, depth: int = 0) -> List[str]:
        """递归拆分文本，优先使用更高级别的分隔符"""
        if len(text) <= max_chars:
            return [text]

        # 安全阀: 深度过高或文本过大时直接强制字符拆分, 防止递归溢出
        if depth >= len(self.separators) or depth > 8 or len(text) > 50000:
            # 达到最底层或超大文本, 强制按字符拆分
            return self._force_split(text, max_chars, overlap)

        separator = self.separators[depth]
        if separator == "":
            return self._force_split(text, max_chars, overlap)

        splits = text.split(separator)
        result = []
        current = ""

        for part in splits:
            candidate = current + (separator if current else "") + part

            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    result.append(current)
                # 如果单个部分仍然太长，递归拆分
                if len(part) > max_chars:
                    sub_splits = self._recursive_split(part, max_chars, overlap, depth + 1)
                    result.extend(sub_splits)
                    current = ""
                else:
                    current = part

        if current:
            result.append(current)

        # 添加重叠（用上一块的尾部）
        if overlap > 0 and len(result) > 1:
            overlapped = []
            for i, chunk in enumerate(result):
                if i > 0:
                    prev_tail = result[i - 1][-overlap:] if len(result[i - 1]) > overlap else result[i - 1]
                    chunk = prev_tail + "\n" + chunk
                overlapped.append(chunk)
            result = overlapped

        return result

    def _force_split(self, text: str, max_chars: int, overlap: int) -> List[str]:
        """强制按最大字符数拆分"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            start = end - overlap
        return chunks


def load_file_metadata(db_path: str, file_hash: str = None, parsed: int = None) -> List[dict]:
    """从 SQLite 读取文件元数据"""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM files WHERE 1=1"
    params = []
    if file_hash:
        query += " AND file_hash = ?"
        params.append(file_hash)
    if parsed is not None:
        query += " AND parsed = ?"
        params.append(parsed)

    cursor.execute(query, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows
