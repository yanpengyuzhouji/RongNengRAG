"""
文件处理器 — 模块化入库核心
接收单个或多个文件路径，执行完整的 parse → chunk → embed → insert 管道

设计原则:
  - 每个文件独立处理，可随时添加/删除/重建索引
  - 不依赖目录扫描，完全由调用方驱动
  - 返回详细的处理报告，便于上层 (API/UI/CLI) 展示进度
"""

import os
import sys
import time
import json
import hashlib
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Callable
from dataclasses import dataclass, field
from enum import Enum

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.pdf_parser import PDFParser
from ingestion.chunker import Chunker, Chunk
from ingestion.embedder import Embedder, create_text_for_embedding
from ingestion.milvus_store import MilvusStore


class FileStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


@dataclass
class ProcessResult:
    """单个文件的处理结果"""
    file_path: str
    file_hash: str
    file_name: str
    status: FileStatus
    chunks_created: int = 0
    chars_extracted: int = 0
    parse_time_ms: float = 0
    embed_time_ms: float = 0
    total_time_ms: float = 0
    error_message: str = ""
    # 元数据
    domain: str = ""
    category: str = ""
    doc_number: str = ""
    file_type: str = ""


@dataclass
class BatchResult:
    """批量处理结果"""
    total: int
    success: int
    failed: int
    results: List[ProcessResult] = field(default_factory=list)
    total_time_ms: float = 0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total > 0 else 0


class FileProcessor:
    """
    模块化文件处理器

    用法:
        processor = FileProcessor()
        result = processor.process("D:/path/to/file.pdf")
        batch_result = processor.process_batch(["file1.pdf", "file2.pdf"])
        processor.delete("file_hash_or_path")
        processor.reindex("file_hash_or_path")
    """

    def __init__(self, config_path: str = None):
        from config import load_config, get_config_path, ensure_data_dirs
        self.config = load_config(config_path)
        self.config_path = config_path or get_config_path()

        self.pdf_parser = PDFParser()
        self.chunker = Chunker(config_path)
        self.embedder = None  # 延迟加载
        self.store = MilvusStore(config_path)

        # 数据库路径 (已由 load_config 解析为绝对路径)
        self.db_path = self.config["paths"]["metadata_db"]
        self.uploads_dir = Path(self.config["paths"]["uploads_dir"])
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        self._init_registry()

    def _init_registry(self):
        """初始化文件注册表 (SQLite)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_registry (
                file_hash TEXT PRIMARY KEY,
                original_path TEXT NOT NULL,
                stored_path TEXT,
                file_name TEXT NOT NULL,
                file_size INTEGER,
                file_type TEXT,
                status TEXT DEFAULT 'pending',
                chunks_count INTEGER DEFAULT 0,
                chars_count INTEGER DEFAULT 0,
                domain TEXT,
                category TEXT,
                doc_number TEXT,
                error_message TEXT,
                parse_time_ms REAL DEFAULT 0,
                embed_time_ms REAL DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                updated_at TEXT DEFAULT (datetime('now', 'localtime')),
                reindex_count INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_registry_status ON file_registry(status)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_registry_domain ON file_registry(domain)
        """)

        conn.commit()
        conn.close()

    def compute_hash(self, filepath: str) -> str:
        """计算文件 SHA256"""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def process(self, file_path: str,
                domain: str = None,
                category: str = None,
                progress_callback: Callable[[str, float], None] = None
                ) -> ProcessResult:
        """
        处理单个文件: parse → chunk → embed → insert

        Args:
            file_path: 文件路径 (绝对路径)
            domain: 手动指定专业域 (可选，默认自动推断)
            category: 手动指定文档类目 (可选)
            progress_callback: 进度回调 (阶段名, 0~1进度)

        Returns:
            ProcessResult
        """
        from ingestion.file_walker import FileWalker

        if not os.path.exists(file_path):
            return ProcessResult(
                file_path=file_path, file_hash="", file_name=Path(file_path).name,
                status=FileStatus.FAILED, error_message=f"文件不存在: {file_path}"
            )

        t_start = time.time()
        file_path_obj = Path(file_path)
        file_name = file_path_obj.name
        file_hash = self.compute_hash(file_path)
        file_ext = file_path_obj.suffix.lower()
        file_size = file_path_obj.stat().st_size

        # 检查是否已入库
        existing = self._get_registry(file_hash)
        if existing and existing["status"] == "completed":
            return ProcessResult(
                file_path=file_path, file_hash=file_hash, file_name=file_name,
                status=FileStatus.COMPLETED,
                chunks_created=existing["chunks_count"],
                chars_extracted=existing["chars_count"],
                domain=existing.get("domain", ""),
                error_message="文件已入库，无需重复处理"
            )

        result = ProcessResult(
            file_path=file_path, file_hash=file_hash, file_name=file_name,
            status=FileStatus.PROCESSING, file_type=file_ext
        )

        # 登记为 processing
        self._upsert_registry(file_hash, file_name, file_path, file_size, file_ext,
                              status="processing")

        try:
            # ===== Step 1: 解析 (25%) =====
            if progress_callback:
                progress_callback("解析文件", 0.0)

            file_meta = self._build_file_meta(file_path, file_hash, domain, category)
            chunks = self._parse_file(file_path, file_meta)
            result.chunks_created = len(chunks)
            result.chars_extracted = sum(c.char_count for c in chunks)
            result.domain = file_meta.get("domain", "")
            result.category = file_meta.get("category", "")
            result.doc_number = file_meta.get("doc_number", "")
            t_parse = (time.time() - t_start) * 1000
            result.parse_time_ms = t_parse

            if not chunks:
                result.status = FileStatus.FAILED
                result.error_message = "解析后无有效文本内容"
                self._upsert_registry(file_hash, file_name, file_path, file_size, file_ext,
                                      status="failed", error=result.error_message)
                return result

            if progress_callback:
                progress_callback("解析文件", 1.0)

            # ===== Step 2: 嵌入 (50%) =====
            if progress_callback:
                progress_callback("生成嵌入向量", 0.0)

            if self.embedder is None:
                self.embedder = Embedder(self.config_path)

            embedding_texts = [create_text_for_embedding(chunk) for chunk in chunks]
            emb_result = self.embedder.encode(embedding_texts, show_progress=False)
            t_embed = (time.time() - t_start) * 1000 - t_parse
            result.embed_time_ms = t_embed

            if progress_callback:
                progress_callback("生成嵌入向量", 1.0)

            # ===== Step 3: 入库 (25%) =====
            if progress_callback:
                progress_callback("写入向量库", 0.0)

            # 先删除旧索引 (如果存在)
            self.store.delete_by_file_hash(file_hash)

            self.store.insert(
                chunks=chunks,
                dense_vectors=emb_result.dense_vectors,
                sparse_vectors=emb_result.sparse_vectors,
                embedding_texts=embedding_texts,
                batch_size=min(500, len(chunks)),
            )

            if progress_callback:
                progress_callback("写入向量库", 1.0)

            # ===== 标记完成 =====
            result.status = FileStatus.COMPLETED
            result.total_time_ms = (time.time() - t_start) * 1000

            self._upsert_registry(
                file_hash, file_name, file_path, file_size, file_ext,
                status="completed",
                chunks_count=len(chunks),
                chars_count=result.chars_extracted,
                domain=result.domain,
                category=result.category,
                doc_number=result.doc_number,
                parse_time=result.parse_time_ms,
                embed_time=result.embed_time_ms,
            )

        except Exception as e:
            result.status = FileStatus.FAILED
            result.error_message = str(e)[:500]
            self._upsert_registry(file_hash, file_name, file_path, file_size, file_ext,
                                  status="failed", error=result.error_message)

        return result

    def process_batch(self, file_paths: List[str],
                      domain: str = None,
                      category: str = None,
                      progress_callback: Callable[[str, float], None] = None
                      ) -> BatchResult:
        """
        批量处理多个文件

        Args:
            file_paths: 文件路径列表
            domain: 统一指定域
            category: 统一指定类目
            progress_callback: 总体进度回调

        Returns:
            BatchResult
        """
        t_start = time.time()
        results = []
        success = 0
        failed = 0

        for i, fp in enumerate(file_paths):
            if progress_callback:
                progress_callback(f"处理中 ({i + 1}/{len(file_paths)})",
                                  i / len(file_paths))

            result = self.process(fp, domain=domain, category=category)
            results.append(result)
            if result.status == FileStatus.COMPLETED:
                success += 1
            else:
                failed += 1

        if progress_callback:
            progress_callback("完成", 1.0)

        return BatchResult(
            total=len(file_paths),
            success=success,
            failed=failed,
            results=results,
            total_time_ms=(time.time() - t_start) * 1000,
        )

    def delete(self, identifier: str) -> bool:
        """
        从向量库中删除文件

        Args:
            identifier: 文件 hash 或文件路径

        Returns:
            是否成功
        """
        file_hash = self._resolve_hash(identifier)
        if not file_hash:
            return False

        # Milvus 删除
        self.store.delete_by_file_hash(file_hash)

        # 注册表标记
        self._upsert_registry(file_hash, status="deleted")
        return True

    def reindex(self, identifier: str,
                progress_callback: Callable = None) -> ProcessResult:
        """
        重建文件索引（先删后加）

        Args:
            identifier: 文件 hash 或文件路径
        """
        file_hash = self._resolve_hash(identifier)
        if not file_hash:
            return ProcessResult(
                file_path=identifier, file_hash="", file_name="",
                status=FileStatus.FAILED, error_message="文件未在注册表中找到"
            )

        reg = self._get_registry(file_hash)
        if not reg:
            return ProcessResult(
                file_path=identifier, file_hash=file_hash, file_name="",
                status=FileStatus.FAILED, error_message="文件注册信息丢失"
            )

        file_path = reg.get("original_path") or reg.get("stored_path")
        if not file_path or not os.path.exists(file_path):
            return ProcessResult(
                file_path=file_path or "", file_hash=file_hash,
                file_name=reg.get("file_name", ""),
                status=FileStatus.FAILED, error_message="原始文件不存在，无法重建索引"
            )

        # 先删后加
        self.delete(file_hash)
        return self.process(file_path, progress_callback=progress_callback)

    def list_files(self, status: str = None, domain: str = None,
                   limit: int = 100, offset: int = 0) -> List[dict]:
        """列出已注册的文件"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM file_registry WHERE 1=1"
        params = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if domain:
            query += " AND domain = ?"
            params.append(domain)

        query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return rows

    def get_summary(self) -> dict:
        """获取索引入库摘要"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT status, COUNT(*) FROM file_registry GROUP BY status")
        by_status = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute("SELECT domain, COUNT(*) FROM file_registry WHERE status='completed' GROUP BY domain")
        by_domain = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute("SELECT SUM(chunks_count), SUM(chars_count) FROM file_registry WHERE status='completed'")
        totals = cursor.fetchone()

        conn.close()

        return {
            "total_files": sum(by_status.values()),
            "by_status": by_status,
            "by_domain": by_domain,
            "total_chunks": totals[0] or 0,
            "total_chars": totals[1] or 0,
        }

    # ===== 内部方法 =====

    def _get_registry(self, file_hash: str) -> Optional[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM file_registry WHERE file_hash = ?", (file_hash,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def _upsert_registry(self, file_hash: str, file_name: str = None,
                         original_path: str = None, file_size: int = None,
                         file_type: str = None, status: str = None,
                         chunks_count: int = None, chars_count: int = None,
                         domain: str = None, category: str = None,
                         doc_number: str = None, error: str = None,
                         parse_time: float = None, embed_time: float = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        existing = self._get_registry(file_hash)

        if existing:
            updates = []
            params = []
            for col, val in [
                ("original_path", original_path), ("file_name", file_name),
                ("file_size", file_size), ("file_type", file_type), ("status", status),
                ("chunks_count", chunks_count), ("chars_count", chars_count),
                ("domain", domain), ("category", category), ("doc_number", doc_number),
                ("error_message", error), ("parse_time_ms", parse_time),
                ("embed_time_ms", embed_time),
            ]:
                if val is not None:
                    updates.append(f"{col} = ?")
                    params.append(val)
            if status == "completed" and existing["status"] == "completed":
                updates.append("reindex_count = reindex_count + 1")
            updates.append("updated_at = datetime('now', 'localtime')")
            params.append(file_hash)
            cursor.execute(f"UPDATE file_registry SET {', '.join(updates)} WHERE file_hash = ?", params)
        else:
            cursor.execute("""
                INSERT INTO file_registry (file_hash, original_path, file_name,
                file_size, file_type, status, chunks_count, chars_count,
                domain, category, doc_number, error_message,
                parse_time_ms, embed_time_ms)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (file_hash, original_path or "", file_name or "", file_size or 0,
                  file_type or "", status or "pending", chunks_count or 0,
                  chars_count or 0, domain or "", category or "",
                  doc_number or "", error or "", parse_time or 0, embed_time or 0))

        conn.commit()
        conn.close()

    def _resolve_hash(self, identifier: str) -> Optional[str]:
        """从文件路径或 hash 解析为 hash"""
        if len(identifier) == 64 and all(c in "0123456789abcdef" for c in identifier):
            return identifier
        if os.path.exists(identifier):
            return self.compute_hash(identifier)
        # 从注册表查找
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT file_hash FROM file_registry WHERE original_path = ? OR file_name = ?",
            (identifier, Path(identifier).name)
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def _build_file_meta(self, file_path: str, file_hash: str,
                         domain: str = None, category: str = None) -> dict:
        """构建元数据字典"""
        from ingestion.file_walker import FileWalker

        fp = Path(file_path)
        # 如果文件在知识库目录中，用 FileWalker 提取元数据
        kb_path = self.config["paths"]["knowledge_base"]
        walker = FileWalker(self.config_path)

        try:
            rel_path = str(fp.relative_to(kb_path))
        except ValueError:
            rel_path = fp.name

        path_meta = walker.extract_path_metadata(rel_path)
        filename_meta = walker.extract_filename_metadata(fp.name, str(fp))

        meta = {
            "file_hash": file_hash,
            "full_path": str(fp),
            "relative_path": rel_path,
            "file_name": fp.name,
            "extension": fp.suffix.lower(),
            "size_bytes": fp.stat().st_size if fp.exists() else 0,
            "domain": domain or path_meta.get("domain") or "",
            "category": category or path_meta.get("category") or "",
            "subcategory": path_meta.get("subcategory") or "",
            "doc_number": filename_meta.get("doc_number") or "",
            "publish_level": filename_meta.get("publish_level") or "",
            "voltage_level": filename_meta.get("voltage_level") or "",
            "discipline": filename_meta.get("discipline") or "",
            "equipment_type": filename_meta.get("equipment_type") or "",
            "year": filename_meta.get("year") or 0,
            "region": filename_meta.get("region") or "全国",
            "drawing_code": filename_meta.get("drawing_code") or "",
        }

        # 判断是否图纸
        is_dwg = meta["extension"] in (".dwg", ".dxf")
        meta["is_drawing"] = 1 if is_dwg else 0
        meta["is_archive"] = 1 if meta["extension"] in (".zip", ".rar", ".7z") else 0
        meta["format_group"] = "drawing" if is_dwg else "document"

        return meta

    def _parse_file(self, file_path: str, file_meta: dict) -> List[Chunk]:
        """解析文件并生成 chunks"""
        ext = file_meta["extension"].lower()
        is_drawing = file_meta.get("is_drawing", 0)

        # PDF
        if ext == ".pdf":
            if is_drawing:
                text = self.pdf_parser.parse_single_page_pdf(file_path) or ""
                return self.chunker.chunk_drawing(text, file_meta)
            else:
                parsed = self.pdf_parser.parse(file_path)
                return self.chunker.chunk_pdf_document(parsed, file_meta)

        # DOC (old binary Word format)
        elif ext == ".doc":
            text = self._parse_doc_file(file_path)
            if text:
                return self.chunker.chunk_text_document(text, file_meta)
            return []

        # DOCX
        elif ext == ".docx":
            try:
                import docx
                doc = docx.Document(file_path)
                text = "\n".join([p.text for p in doc.paragraphs])
                return self.chunker.chunk_text_document(text, file_meta)
            except Exception:
                return []

        # TXT / MD
        elif ext in (".txt", ".md"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return self.chunker.chunk_text_document(text, file_meta)

        # OFD
        elif ext == ".ofd":
            try:
                from ofdparser import OFDParser
                parser = OFDParser()
                ofd_doc = parser.parse(file_path)
                all_text = []
                for page in ofd_doc.pages:
                    page_texts = []
                    for elem in page.elements:
                        if hasattr(elem, 'text') and elem.text:
                            page_texts.append(elem.text)
                    all_text.append("\n".join(page_texts))
                text = "\n\n".join(all_text)
                return self.chunker.chunk_text_document(text, file_meta)
            except ImportError:
                return []
            except Exception:
                return []

        # XLSX
        elif ext in (".xls", ".xlsx"):
            try:
                import openpyxl
                wb = openpyxl.load_workbook(file_path, data_only=True)
                texts = []
                for sheet_name in wb.sheetnames:
                    ws = wb[sheet_name]
                    sheet_text = [f"[Sheet: {sheet_name}]"]
                    for row in ws.iter_rows(values_only=True):
                        row_text = " | ".join([str(c) if c is not None else "" for c in row])
                        if row_text.strip(" |"):
                            sheet_text.append(row_text)
                    texts.append("\n".join(sheet_text))
                text = "\n\n".join(texts)
                return self.chunker.chunk_text_document(text, file_meta)
            except Exception:
                return []

        # DWG (跳过，需专用解析器)
        elif ext in (".dwg", ".dxf"):
            return []

        # 其他
        else:
            return []

    def _parse_doc_file(self, file_path: str) -> str:
        """
        解析旧版 .doc 文件 (OLE2 复合文档格式)
        按优先级尝试多种后端:
          1. win32com (Windows MS Word COM 自动化, 最可靠)
          2. LibreOffice headless 转换
          3. olefile 原始文本提取
          4. antiword (Linux)
          5. docx2txt / python-docx (仅对伪装的 .docx 有效)
        """
        # 方案1: Windows COM (MS Word 安装时最可靠)
        text = self._parse_doc_via_win32(file_path)
        if text and text.strip():
            return text

        # 方案2: LibreOffice headless 转换
        text = self._parse_doc_via_libreoffice(file_path)
        if text and text.strip():
            return text

        # 方案3: olefile 原始提取
        text = self._parse_doc_via_olefile(file_path)
        if text and text.strip():
            return text

        # 方案4: antiword (Linux)
        text = self._parse_doc_via_antiword(file_path)
        if text and text.strip():
            return text

        # 方案5: docx2txt / python-docx (某些 .doc 实际是 .docx 改名)
        try:
            import docx2txt
            text = docx2txt.process(file_path)
            if text and text.strip():
                print(f"   [doc] docx2txt 解析成功: {len(text)} 字符")
                return text
        except Exception:
            pass

        try:
            import docx
            doc = docx.Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
            if text and text.strip():
                print(f"   [doc] python-docx 解析成功: {len(text)} 字符")
                return text
        except Exception:
            pass

        print(f"   [warn] 所有 .doc 解析方案均失败: {os.path.basename(file_path)}")
        print(f"   [tip] 建议方案: (1) pip install pywin32 启用 Word COM 解析")
        print(f"         或 (2) 安装 LibreOffice")
        print(f"         或 (3) 用 Word 打开后另存为 .docx 格式")
        return ""

    def _parse_doc_via_win32(self, file_path: str) -> str:
        """
        通过 Windows COM 调用 Microsoft Word 提取文本
        这是 Windows 上解析 .doc 最可靠的方式
        """
        try:
            import pythoncom
            import win32com.client
        except ImportError:
            return ""

        abs_path = os.path.abspath(file_path)
        word = None
        doc = None
        try:
            pythoncom.CoInitialize()
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            word.DisplayAlerts = 0

            # 打开文档
            doc = word.Documents.Open(abs_path, ReadOnly=True, Visible=False)

            # 提取所有文本
            text = doc.Content.Text

            # 关闭文档
            doc.Close(SaveChanges=False)

            if text and text.strip():
                print(f"   [doc] Word COM 解析成功: {len(text)} 字符")
                return text

        except Exception as e:
            print(f"   [doc] Word COM 失败: {e}")
            # 确保即使出错也尝试关闭文档
            if doc is not None:
                try:
                    doc.Close(SaveChanges=False)
                except Exception:
                    pass
        finally:
            if word is not None:
                try:
                    word.Quit()
                except Exception:
                    pass
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

        return ""

    def _parse_doc_via_libreoffice(self, file_path: str) -> str:
        """通过 LibreOffice headless 将 .doc 转为文本"""
        import subprocess
        import tempfile

        # 查找 LibreOffice 路径
        lo_paths = [
            "libreoffice", "soffice",
            "C:\\Program Files\\LibreOffice\\program\\soffice.exe",
            "C:\\Program Files (x86)\\LibreOffice\\program\\soffice.exe",
            "/usr/bin/libreoffice", "/usr/bin/soffice",
        ]

        lo_exe = None
        for p in lo_paths:
            try:
                subprocess.run([p, "--version"], capture_output=True, timeout=5)
                lo_exe = p
                break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        if not lo_exe:
            return ""

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cmd = [
                    lo_exe, "--headless", "--convert-to", "txt:Text",
                    "--outdir", tmpdir, file_path,
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=60)
                if result.returncode == 0:
                    # 查找生成的 txt 文件
                    for f in os.listdir(tmpdir):
                        if f.endswith(".txt"):
                            txt_path = os.path.join(tmpdir, f)
                            with open(txt_path, "r", encoding="utf-8", errors="ignore") as fp:
                                text = fp.read()
                            if text.strip():
                                print(f"   [doc] LibreOffice 解析成功: {len(text)} 字符")
                                return text
        except Exception:
            pass

        return ""

    def _parse_doc_via_olefile(self, file_path: str) -> str:
        """通过 olefile 从 OLE2 容器中提取原始文本"""
        try:
            import olefile
            ole = olefile.OleFileIO(file_path)

            # 尝试读取 WordDocument 流中的文本
            text_parts = []

            # 读取主文本流
            if ole.exists("WordDocument"):
                data = ole.openstream("WordDocument").read()
                # 尝试提取可读文本 (UTF-16 LE 编码的文本片段)
                try:
                    decoded = data.decode("utf-16-le", errors="ignore")
                    # 过滤控制字符，保留可读内容
                    import re
                    readable = re.findall(r'[一-鿿　-〿＀-￯a-zA-Z0-9\s.,;:!?()（）、。，；：！？""''【】《》/-]+', decoded)
                    if readable:
                        text_parts.extend(readable)
                except Exception:
                    pass

            # 尝试 1Table 或 0Table 流
            for stream_name in ole.listdir():
                stream_path = "/".join(stream_name) if isinstance(stream_name, list) else stream_name
                if "Table" in stream_path or "Text" in stream_path:
                    try:
                        data = ole.openstream(stream_path).read()
                        decoded = data.decode("utf-16-le", errors="ignore")
                        import re
                        readable = re.findall(r'[一-鿿]+', decoded)
                        if readable:
                            text_parts.extend(readable)
                    except Exception:
                        pass

            ole.close()

            if text_parts:
                text = " ".join(text_parts)
                print(f"   [doc] olefile 解析成功: {len(text)} 字符 (可能不完整)")
                return text
        except ImportError:
            pass
        except Exception:
            pass

        return ""

    def _parse_doc_via_antiword(self, file_path: str) -> str:
        """通过 antiword 解析 (Linux)"""
        import subprocess
        try:
            result = subprocess.run(
                ["antiword", file_path],
                capture_output=True, timeout=30,
            )
            if result.returncode == 0:
                text = result.stdout.decode("utf-8", errors="ignore")
                if text.strip():
                    print(f"   [doc] antiword 解析成功: {len(text)} 字符")
                    return text
        except FileNotFoundError:
            pass
        except Exception:
            pass
        return ""
