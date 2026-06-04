"""
PDF 解析器 — 使用 PyMuPDF (fitz) 提取文本
支持单页 CAD 导出 PDF 和多页标准文档
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional
import re


class PDFParser:
    """PDF 文本提取器，对扫描件标记 OCR 需求"""

    def __init__(self, min_text_chars: int = 50):
        """
        Args:
            min_text_chars: 单页最少文本字符数，低于此值标记为扫描件需 OCR
        """
        self.min_text_chars = min_text_chars

    def parse(self, filepath: str) -> dict:
        """
        解析 PDF 文件
        返回:
            {
                "pages": [{"page_num": 1, "text": "...", "char_count": 150, "needs_ocr": false}],
                "metadata": {"title": "...", "author": "...", "subject": "..."},
                "page_count": 10,
                "total_chars": 5000,
                "needs_ocr_pages": [3, 5],  # 需要 OCR 的页码
                "is_scanned": false
            }
        """
        doc = fitz.open(filepath)
        result = {
            "pages": [],
            "metadata": dict(doc.metadata) if doc.metadata else {},
            "page_count": len(doc),
            "total_chars": 0,
            "needs_ocr_pages": [],
            "is_scanned": False,
        }

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            char_count = len(text.strip())

            page_data = {
                "page_num": page_num + 1,
                "text": text.strip(),
                "char_count": char_count,
                "needs_ocr": char_count < self.min_text_chars,
            }
            result["pages"].append(page_data)
            result["total_chars"] += char_count

            if page_data["needs_ocr"]:
                result["needs_ocr_pages"].append(page_num + 1)

        # 如果超过 50% 的页面需要 OCR，标记为扫描文档
        if len(result["needs_ocr_pages"]) > len(doc) * 0.5:
            result["is_scanned"] = True

        doc.close()
        return result

    def parse_single_page_pdf(self, filepath: str) -> Optional[str]:
        """
        快速解析单页 PDF（用于 CAD 导出图纸）
        大部分 CAD PDF 文本稀疏，主要依赖元数据检索
        """
        doc = fitz.open(filepath)
        if len(doc) == 0:
            doc.close()
            return None

        # 单页 PDF：提取所有文本
        text = ""
        for page in doc:
            text += page.get_text("text")

        doc.close()
        return text.strip()

    @staticmethod
    def extract_drawing_text_with_layout(filepath: str) -> List[dict]:
        """
        按位置提取图纸文本块（保留空间关系）
        用于后续结构化处理
        """
        doc = fitz.open(filepath)
        blocks = []

        for page in doc:
            text_blocks = page.get_text("blocks")
            for block in text_blocks:
                x0, y0, x1, y1, text, block_type, block_no = block
                if text.strip():
                    blocks.append({
                        "page": page.number + 1,
                        "bbox": (x0, y0, x1, y1),
                        "text": text.strip(),
                        "block_type": block_type,
                    })

        doc.close()
        return blocks
