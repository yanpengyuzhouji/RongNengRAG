"""
PDF 解析器 — 使用 PyMuPDF (fitz) 提取文本
支持单页 CAD 导出 PDF 和多页标准文档
支持 PaddleOCR 扫描件识别 (子进程隔离)
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional
import re


class PDFParser:
    """PDF 文本提取器，内置 OCR 能力处理扫描件"""

    def __init__(self, min_text_chars: int = 50,
                 ocr_config: dict = None):
        """
        Args:
            min_text_chars: 单页最少文本字符数，低于此值标记为扫描件需 OCR
            ocr_config: OCR 配置 {"enabled": bool, "dpi": int, "lang": str, ...}
        """
        self.min_text_chars = min_text_chars
        self.ocr_config = ocr_config or {}
        self._ocr_engine = None

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

    # ===== OCR 扫描件识别 =====

    def _get_ocr_engine(self):
        """延迟加载 OCR 引擎 (子进程隔离模式)"""
        if self._ocr_engine is None and self.ocr_config.get("enabled"):
            from ingestion.ocr_engine import OCREngine
            self._ocr_engine = OCREngine(
                lang=self.ocr_config.get("lang", "ch"),
                dpi=self.ocr_config.get("dpi", 200),
                use_subprocess=self.ocr_config.get("use_subprocess", True),
            )
        return self._ocr_engine

    def ocr_pages(self, filepath: str,
                  pages: List[int]) -> Dict[int, str]:
        """
        对 PDF 中指定的页面执行 OCR 识别

        Args:
            filepath: PDF 文件路径
            pages: 需要 OCR 的页面索引列表 (0-based)

        Returns:
            {page_index: "识别文本", ...}
        """
        engine = self._get_ocr_engine()
        if engine is None:
            return {}

        result = engine.ocr_pdf_pages(filepath, pages)

        if not result.get("success"):
            print(f"   [OCR] 识别失败: {result.get('error', '未知')}")
            return {}

        page_texts = {}
        pages_data = result.get("pages", {})
        for page_str, page_data in pages_data.items():
            page_num = int(page_str)
            text = page_data.get("text", "")
            if text.strip():
                page_texts[page_num - 1] = text  # 转回 0-based

        elapsed = result.get("elapsed_ms", 0)
        if elapsed > 0:
            print(f"   [OCR] {len(pages)} 页扫描件识别完成, "
                  f"耗时 {elapsed:.0f}ms ({elapsed / len(pages):.0f}ms/页)")

        return page_texts

    def ocr_page(self, image_path: str) -> str:
        """
        对单张图片执行 OCR

        Args:
            image_path: 图片文件路径

        Returns:
            识别文本
        """
        engine = self._get_ocr_engine()
        if engine is None:
            return ""

        result = engine.ocr_page(image_path)
        if result.get("success"):
            return result.get("text", "")
        return ""
