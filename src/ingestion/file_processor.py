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

        ocr_cfg = self.config.get("ocr", {})
        self.pdf_parser = PDFParser(
            min_text_chars=ocr_cfg.get("min_text_chars", 50),
            ocr_config=ocr_cfg,
        )
        self.chunker = Chunker(config_path)
        self.embedder = None  # 延迟加载
        self.store = MilvusStore(config_path)

        # 数据库路径 (已由 load_config 解析为绝对路径)
        self.db_path = self.config["paths"]["metadata_db"]
        self.uploads_dir = Path(self.config["paths"]["uploads_dir"])
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        self._init_registry()

    def _get_db_connection(self):
        """获取 SQLite 连接，统一启用 WAL 模式 + busy_timeout，避免 database is locked"""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_registry(self):
        """初始化文件注册表 (SQLite)"""
        conn = self._get_db_connection()
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

        self._recover_stuck_files()

    def _recover_stuck_files(self):
        """启动时恢复卡在 processing 状态的文件 (上次崩溃残留)"""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE file_registry SET status='pending', error_message='' "
            "WHERE status='processing'"
        )
        n = cursor.rowcount
        if n > 0:
            print(f"[recover] 恢复 {n} 个卡在 processing 状态的文件 -> pending")
        conn.commit()
        conn.close()

    def _cleanup_ocr_temp(self):
        """入库完成后清理 OCR 临时目录 (防止大量 PNG 堆积)"""
        try:
            ocr_tmp = self.config.get("ocr", {}).get(
                "ocr_tmp_dir", "E:/RongNengRAG/data/ocr_tmp"
            )
            if os.path.isdir(ocr_tmp):
                import shutil
                for name in os.listdir(ocr_tmp):
                    path = os.path.join(ocr_tmp, name)
                    if os.path.isdir(path) and name.startswith("ocr_"):
                        shutil.rmtree(path, ignore_errors=True)
                    elif os.path.isfile(path) and name.startswith("ocr_"):
                        try:
                            os.remove(path)
                        except OSError:
                            pass
        except Exception:
            pass  # 清理失败不影响主流程

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
                progress_callback: Callable[[str, float], None] = None,
                turbo: bool = None,
                turbo_max_workers: int = None,
                ) -> ProcessResult:
        """
        处理单个文件: parse → chunk → embed → insert

        Args:
            file_path: 文件路径 (绝对路径)
            domain: 手动指定专业域 (可选，默认自动推断)
            category: 手动指定文档类目 (可选)
            progress_callback: 进度回调 (阶段名, 0~1进度)
            turbo: 覆盖 config.yaml 的 turbo.enabled (None=使用配置文件)
            turbo_max_workers: 覆盖 max_workers (None=使用配置文件)

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

        # Turbo 覆盖: CLI --turbo/--max-workers 运行时覆盖 config.yaml
        if turbo is not None or turbo_max_workers is not None:
            turbo_cfg = self.config.setdefault("ocr", {}).setdefault("turbo", {})
            if turbo is not None:
                turbo_cfg["enabled"] = turbo
            if turbo_max_workers is not None:
                turbo_cfg["max_workers"] = turbo_max_workers
            # 同步到 pdf_parser.ocr_config (已缓存的 _ocr_engine 不受影响)
            self.pdf_parser.ocr_config.setdefault("turbo", {}).update(turbo_cfg)

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
            # ===== Step 1: 解析 =====
            if progress_callback:
                progress_callback("解析文件", 0.0)

            file_meta = self._build_file_meta(file_path, file_hash, domain, category)
            file_ext_lower = file_meta.get("extension", "").lower()

            # PDF + OCR 场景: 渐进入库，避免全部OCR完才嵌入
            if file_ext_lower == ".pdf" and not file_meta.get("is_drawing"):
                ocr_cfg = self.config.get("ocr", {})
                if ocr_cfg.get("enabled"):
                    progressive_result = self._process_pdf_progressive(
                        file_path, file_meta, file_hash, file_name,
                        file_size, file_ext, result, progress_callback
                    )
                    if progressive_result is not None:
                        return progressive_result  # 已处理完成（成功或失败）

            # 通用路径: 解析 → 嵌入 → 入库
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

            # ===== Step 2: 嵌入 =====
            if progress_callback:
                progress_callback("生成嵌入向量", 0.0)

            result = self._embed_and_insert(
                chunks, result, file_hash, file_name, file_path,
                file_size, file_ext, t_start, progress_callback
            )
            return result

        except MemoryError:
            result.status = FileStatus.FAILED
            result.error_message = "内存不足: 文档过大导致chunker递归溢出, 请尝试拆分文件后重新入库"
            self._upsert_registry(file_hash, file_name, file_path, file_size, file_ext,
                                  status="failed", error=result.error_message)
        except Exception as e:
            result.status = FileStatus.FAILED
            result.error_message = str(e) or type(e).__name__
            if not result.error_message.strip():
                result.error_message = f"未知错误: {type(e).__name__}"
            result.error_message = result.error_message[:500]
            self._upsert_registry(file_hash, file_name, file_path, file_size, file_ext,
                                  status="failed", error=result.error_message)

        return result

    def process_batch(self, file_paths: List[str],
                      domain: str = None,
                      category: str = None,
                      progress_callback: Callable[[str, float], None] = None,
                      turbo: bool = None,
                      turbo_max_workers: int = None,
                      ) -> BatchResult:
        """
        批量处理多个文件

        Args:
            file_paths: 文件路径列表
            domain: 统一指定域
            category: 统一指定类目
            progress_callback: 总体进度回调
            turbo: 覆盖 config.yaml 的 turbo.enabled
            turbo_max_workers: 覆盖 max_workers

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

            result = self.process(fp, domain=domain, category=category,
                                  turbo=turbo, turbo_max_workers=turbo_max_workers)
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

    def delete(self, identifier: str, remove_file: bool = False) -> bool:
        """
        从向量库中删除文件

        Args:
            identifier: 文件 hash 或文件路径
            remove_file: 是否同时删除物理文件（仅限 uploads 目录下的文件）

        Returns:
            是否成功
        """
        file_hash = self._resolve_hash(identifier)
        if not file_hash:
            return False

        # 获取注册表信息（用于后续清理物理文件）
        reg = self._get_registry(file_hash)

        # Milvus 删除
        self.store.delete_by_file_hash(file_hash)

        # 清理物理文件（安全策略：仅删除 uploads 目录下的文件）
        if remove_file and reg:
            file_path = reg.get("original_path") or reg.get("stored_path") or ""
            if file_path and os.path.exists(file_path):
                try:
                    uploads_abs = str(self.uploads_dir.resolve())
                    file_abs = str(Path(file_path).resolve())
                    if file_abs.startswith(uploads_abs):
                        os.remove(file_path)
                except Exception:
                    pass

        # 注册表标记
        self._upsert_registry(file_hash, status="deleted")
        return True

    def sync_orphans(self, dry_run: bool = False) -> dict:
        """
        扫描注册表中指向不存在物理文件的孤记录，清理向量 + 标记 deleted

        Args:
            dry_run: True 只扫描不清理

        Returns:
            {total_checked, orphan_count, cleaned, errors, orphans (仅 dry_run)}
        """
        conn = self._get_db_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM file_registry WHERE status NOT IN ('deleted', 'pending', 'processing')"
        )
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        orphans = []
        for row in rows:
            file_path = row.get("original_path") or row.get("stored_path") or ""
            if not file_path or not os.path.exists(file_path):
                orphans.append(row)

        if dry_run:
            return {
                "total_checked": len(rows),
                "orphan_count": len(orphans),
                "orphans": [
                    {
                        "file_name": o.get("file_name", ""),
                        "file_hash": o.get("file_hash", ""),
                        "original_path": o.get("original_path", "") or o.get("stored_path", ""),
                        "chunks_count": o.get("chunks_count", 0),
                        "domain": o.get("domain", ""),
                        "category": o.get("category", ""),
                    }
                    for o in orphans
                ],
                "cleaned": 0,
            }

        cleaned = 0
        errors = []
        for row in orphans:
            file_hash = row.get("file_hash")
            file_name = row.get("file_name", "")
            try:
                self.store.delete_by_file_hash(file_hash)
                self._upsert_registry(file_hash, status="deleted")
                cleaned += 1
            except Exception as e:
                errors.append({
                    "file_name": file_name,
                    "file_hash": file_hash,
                    "error": str(e),
                })

        return {
            "total_checked": len(rows),
            "orphan_count": len(orphans),
            "cleaned": cleaned,
            "errors": errors,
        }

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
                   limit: int = 100, offset: int = 0,
                   check_existence: bool = True) -> List[dict]:
        """列出已注册的文件

        Args:
            check_existence: 检查物理文件是否存在，添加 file_exists 字段
        """
        conn = self._get_db_connection()
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

        # 检测物理文件是否存在
        for row in rows:
            file_path = (row.get("original_path") or row.get("stored_path") or "")
            if file_path and os.path.exists(file_path):
                row["file_exists"] = True
            elif file_path and not os.path.exists(file_path):
                row["file_exists"] = False
            else:
                row["file_exists"] = None  # 无路径信息

        return rows

    def get_summary(self) -> dict:
        """获取索引入库摘要"""
        conn = self._get_db_connection()
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
        conn = self._get_db_connection()
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
        conn = self._get_db_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 内联 SELECT，避免 _get_registry 开第二个连接导致 database is locked
        cursor.execute("SELECT * FROM file_registry WHERE file_hash = ?", (file_hash,))
        row = cursor.fetchone()
        existing = dict(row) if row else None

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
            # 成功后清除旧错误信息
            if status == "completed":
                updates.append("error_message = ''")
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

    def _process_pdf_progressive(self, file_path, file_meta, file_hash, file_name,
                                  file_size, file_ext, result: ProcessResult,
                                  progress_callback=None) -> Optional[ProcessResult]:
        """
        PDF 渐进入库 — 显式 GPU 模型生命期管理

        流程:
          1. 预解析 + 渲染非OCR页文本
          2. 卸载 BGE-M3 释放 ~2GB 显存
          3. OCR (单进程 safe / 进程池 turbo) → GPU峰值 ~2.5GB(safe) ~8GB(turbo)
          4. OCR子进程退出 → 显存释放
          5. 重新加载 BGE-M3 (~2GB)
          6. 分块 + 嵌入 + 写入 Milvus

        关键约束:
          - OCR子进程使用 PPOCRLabel venv 独立CUDA context
          - BGE-M3 卸载后 WDDM 将其缓存标记为可回收 → OCR子进程可复用
          - 全局 _ocr_lock 保证同一时刻只运行一个 OCR 池
        """
        parsed = self.pdf_parser.parse(file_path)
        needs_ocr = parsed.get("needs_ocr_pages", [])
        if not needs_ocr:
            return None

        ocr_pages_0based = [p - 1 for p in needs_ocr]
        t_start = time.time()

        # 收集非OCR页文本 (CPU, 快速)
        non_ocr_pages = [p for p in parsed.get("pages", [])
                         if not p.get("needs_ocr")]
        all_chunks = []
        total_chars = 0
        if non_ocr_pages:
            non_ocr_text = "\n\n".join(p.get("text", "") for p in non_ocr_pages)
            if non_ocr_text.strip():
                nc = self.chunker.chunk_text_document(non_ocr_text, file_meta)
                all_chunks.extend(nc)
                total_chars += sum(c.char_count for c in nc)

        # 清理旧索引
        self.store.delete_by_file_hash(file_hash)

        # ===== 卸载 BGE-M3 → 释放显存供 OCR 使用 =====
        if self.embedder is not None:
            try:
                from src.utils.gpu_monitor import log_vram_snapshot
                log_vram_snapshot("BGE-M3卸载前")
            except Exception:
                pass
            self.embedder.unload()
            try:
                from src.utils.gpu_monitor import log_vram_snapshot
                log_vram_snapshot("BGE-M3卸载后")
            except Exception:
                pass
        else:
            try:
                from src.utils.gpu_monitor import log_vram_snapshot
                log_vram_snapshot("OCR前(BGE-M3未加载)")
            except Exception:
                pass

        try:
            # ===== OCR — 单进程流式 (--stream 长连接) =====
            if progress_callback:
                progress_callback(
                    f"OCR识别 ({len(ocr_pages_0based)}页)", 0.1
                )

            print(f"   [OCR] 开始识别 {len(ocr_pages_0based)} 页 (流式单进程)...")
            ocr_texts = self.pdf_parser.ocr_pages(file_path, ocr_pages_0based)

            if progress_callback:
                progress_callback("OCR完成, 分块中...", 0.5)

            # 分块OCR结果
            if ocr_texts:
                ocr_text = "\n\n".join(ocr_texts.values())
                if ocr_text.strip():
                    ocr_chunks = self.chunker.chunk_text_document(ocr_text, file_meta)
                    all_chunks.extend(ocr_chunks)
                    total_chars += sum(c.char_count for c in ocr_chunks)

            ocr_elapsed = time.time() - t_start
            print(f"   [OCR] 完成: {len(ocr_texts)}有效页 → "
                  f"{len(all_chunks)} chunks ({ocr_elapsed:.0f}s)")

            # ===== OCR子进程已退出，重新加载 BGE-M3 → 嵌入 =====
            if progress_callback:
                progress_callback("嵌入向量", 0.7)

            embed_total_ms = 0.0
            if all_chunks:
                # 重新加载 BGE-M3 (OCR子进程已释放显存)
                if self.embedder is None:
                    self.embedder = Embedder(self.config_path)
                else:
                    self.embedder.reload()

                try:
                    from src.utils.gpu_monitor import log_vram_snapshot
                    log_vram_snapshot("BGE-M3重载后")
                except Exception:
                    pass

                t_embed = time.time()
                embedding_texts = [create_text_for_embedding(c)
                                   for c in all_chunks]
                emb_result = self.embedder.encode(embedding_texts,
                                                   show_progress=False)
                embed_total_ms = (time.time() - t_embed) * 1000

                if progress_callback:
                    progress_callback("写入向量库", 0.9)

                self.store.insert(
                    chunks=all_chunks,
                    dense_vectors=emb_result.dense_vectors,
                    sparse_vectors=emb_result.sparse_vectors,
                    embedding_texts=embedding_texts,
                    batch_size=min(500, len(all_chunks)),
                )

        except Exception as e:
            print(f"   [ERR] 入库中断: {e}")
            self.store.delete_by_file_hash(file_hash)
            raise

        # 标记完成
        if not all_chunks:
            result.status = FileStatus.FAILED
            result.error_message = "解析后无有效文本内容 (OCR全部失败)"
            self._upsert_registry(file_hash, file_name, file_path, file_size,
                                  file_ext, status="failed",
                                  error=result.error_message)
            return result

        result.chunks_created = len(all_chunks)
        result.chars_extracted = total_chars
        result.domain = file_meta.get("domain", "")
        result.category = file_meta.get("category", "")
        result.doc_number = file_meta.get("doc_number", "")
        result.parse_time_ms = (time.time() - t_start) * 1000
        result.embed_time_ms = embed_total_ms
        result.status = FileStatus.COMPLETED
        result.total_time_ms = (time.time() - t_start) * 1000

        self._upsert_registry(
            file_hash, file_name, file_path, file_size, file_ext,
            status="completed", chunks_count=result.chunks_created,
            chars_count=result.chars_extracted, domain=result.domain,
            category=result.category, doc_number=result.doc_number,
            parse_time=result.parse_time_ms, embed_time=result.embed_time_ms,
        )

        # 及时清理 OCR 临时目录 (防止 E 盘被大量 PNG 占满)
        self._cleanup_ocr_temp()

        return result

    def _embed_and_insert(self, chunks, result: ProcessResult,
                          file_hash, file_name, file_path, file_size, file_ext,
                          t_start, progress_callback=None) -> ProcessResult:
        """嵌入 + 入库 + 标记完成"""
        if self.embedder is None:
            self._wait_for_gpu_slot("BGE-M3 嵌入模型加载")
            self.embedder = Embedder(self.config_path)

        if progress_callback:
            progress_callback("生成嵌入向量", 0.0)

        t_embed = time.time()
        embedding_texts = [create_text_for_embedding(chunk) for chunk in chunks]
        emb_result = self.embedder.encode(embedding_texts, show_progress=False)
        result.embed_time_ms = (time.time() - t_embed) * 1000

        if progress_callback:
            progress_callback("生成嵌入向量", 1.0)

        if progress_callback:
            progress_callback("写入向量库", 0.0)

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
        return result

    def _wait_for_gpu_slot(self, task_name: str = "入库", min_free_mb: int = 2500):
        """GPU 显存背压 — 等待足够显存后再执行操作"""
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from utils.gpu_monitor import get_gpu_monitor
            monitor = get_gpu_monitor(min_free_vram_mb=min_free_mb)
            ok = monitor.wait_for_vram(min_free_mb=min_free_mb)
            if not ok:
                print(f"   [WARN] [{task_name}] 显存等待超时，强制执行")
            return ok
        except ImportError:
            pass
        except Exception:
            pass
        return True

    def _resolve_hash(self, identifier: str) -> Optional[str]:
        """从文件路径或 hash 解析为 hash"""
        if len(identifier) == 64 and all(c in "0123456789abcdef" for c in identifier):
            return identifier
        if os.path.exists(identifier):
            return self.compute_hash(identifier)
        # 从注册表查找
        conn = self._get_db_connection()
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

                # === OCR 集成: 对扫描件页面进行识别 ===
                ocr_enabled = self.config.get("ocr", {}).get("enabled", False)
                needs_ocr = parsed.get("needs_ocr_pages", [])

                if ocr_enabled and needs_ocr:
                    # 将 OCR 页面索引转为 0-based
                    ocr_pages_0based = [p - 1 for p in needs_ocr]
                    print(f"   [OCR] 检测到 {len(needs_ocr)} 页需要 OCR: "
                          f"{needs_ocr[:10]}{'...'if len(needs_ocr)>10 else ''}")

                    # 卸载 BGE-M3 释放显存供 OCR 使用
                    if self.embedder is not None:
                        self.embedder.unload()

                    try:
                        ocr_texts = self.pdf_parser.ocr_pages(
                            file_path, ocr_pages_0based
                        )
                        # 将 OCR 结果回填到 parsed 数据中
                        for page_idx, ocr_text in ocr_texts.items():
                            if page_idx < len(parsed["pages"]):
                                old_text = parsed["pages"][page_idx]["text"]
                                old_chars = parsed["pages"][page_idx]["char_count"]
                                parsed["pages"][page_idx]["text"] = ocr_text
                                parsed["pages"][page_idx]["char_count"] = len(ocr_text)
                                parsed["pages"][page_idx]["needs_ocr"] = False
                                parsed["pages"][page_idx]["ocr_applied"] = True
                                parsed["total_chars"] += len(ocr_text) - old_chars
                                if old_text.strip():
                                    # 原有一些文本，合并而非替换
                                    parsed["pages"][page_idx]["text"] = (
                                        old_text + "\n" + ocr_text
                                    )
                                    parsed["pages"][page_idx]["char_count"] = (
                                        old_chars + len(ocr_text)
                                    )

                        # 更新 needs_ocr_pages
                        still_needs = [
                            p for p in needs_ocr
                            if (p - 1) not in ocr_texts
                        ]
                        parsed["needs_ocr_pages"] = still_needs
                        parsed["is_scanned"] = len(still_needs) > len(needs_ocr) * 0.5

                    except Exception as exc:
                        print(f"   [OCR] OCR 失败 (跳过): {exc}")

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

                # 提取段落文本
                para_text = "\n".join([p.text for p in doc.paragraphs])

                # 提取表格内容（电力规范docx大量信息在表格中）
                table_parts = []
                for ti, table in enumerate(doc.tables):
                    rows_text = []
                    for row in table.rows:
                        cells = []
                        for cell in row.cells:
                            cell_text = " ".join(p.text for p in cell.paragraphs if p.text.strip())
                            if cell_text:
                                cells.append(cell_text)
                        if cells:
                            rows_text.append(" | ".join(cells))
                    if rows_text:
                        table_parts.append(
                            f"[表格{ti + 1}]\n" + "\n".join(rows_text)
                        )

                text = para_text
                if table_parts:
                    text += "\n\n" + "\n\n".join(table_parts)

                if not text.strip():
                    return []

                return self.chunker.chunk_text_document(text, file_meta)

            except ImportError:
                print(f"   [warn] python-docx 未安装，无法解析 .docx: {file_path}")
                return []
            except Exception as e:
                print(f"   [warn] .docx 解析失败: {Path(file_path).name}: {e}")
                return []

        # TXT / MD
        elif ext in (".txt", ".md"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return self.chunker.chunk_text_document(text, file_meta)

        # OFD
        elif ext == ".ofd":
            text = self._parse_ofd_file(file_path)
            if not text or not text.strip():
                return []
            # 超大文本分批: 超过100K字符分批处理, 防止chunker内存溢出
            if len(text) > 100000:
                # 主分隔符: 双换行 (页面间), 页内段落间
                sections = [s.strip() for s in text.split("\n\n") if s.strip()]
                # 如果双换行无效 (只有1段超大文本), 退回到单换行分句
                if len(sections) == 1 and len(sections[0]) > 30000:
                    sections = [s.strip() for s in text.split("\n") if s.strip()]
                all_chunks = []
                batch = ""
                for sec in sections:
                    if len(batch) + len(sec) < 30000:
                        batch = (batch + "\n\n" + sec) if batch else sec
                    else:
                        if batch:
                            try:
                                all_chunks.extend(
                                    self.chunker.chunk_text_document(batch, file_meta)
                                )
                            except MemoryError:
                                # 内存不足, 强制按字符拆分
                                for i in range(0, len(batch), 2000):
                                    sub = batch[i:i+2000].strip()
                                    if sub:
                                        try:
                                            all_chunks.extend(
                                                self.chunker.chunk_text_document(sub, file_meta)
                                            )
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                        batch = sec
                if batch:
                    try:
                        all_chunks.extend(
                            self.chunker.chunk_text_document(batch, file_meta)
                        )
                    except Exception:
                        pass
                return all_chunks
            return self.chunker.chunk_text_document(text, file_meta)

        # XLSX (Office Open XML)
        elif ext == ".xlsx":
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
            except Exception as e:
                print(f"[warn] XLSX解析失败 ({os.path.basename(file_path)}): {e}")
                return []

        # XLS (OLE2 binary format — openpyxl can't handle this)
        elif ext == ".xls":
            try:
                import xlrd
                wb = xlrd.open_workbook(file_path)
                texts = []
                for sheet_name in wb.sheet_names():
                    ws = wb.sheet_by_name(sheet_name)
                    sheet_text = [f"[Sheet: {sheet_name}]"]
                    for row_idx in range(ws.nrows):
                        row_values = ws.row_values(row_idx)
                        row_text = " | ".join([str(c) if c != "" else "" for c in row_values])
                        if row_text.strip(" |"):
                            sheet_text.append(row_text)
                    texts.append("\n".join(sheet_text))
                text = "\n\n".join(texts)
                return self.chunker.chunk_text_document(text, file_meta)
            except Exception as e:
                print(f"[warn] XLS解析失败 ({os.path.basename(file_path)}): {e}")
                return []

        # WPS (金山 WPS 文字文档)
        # WPS 有两代格式: 旧版 OLE2 二进制容器, 新版 ZIP+XML 容器
        elif ext == ".wps":
            text = self._parse_wps_file(file_path)
            if text:
                return self.chunker.chunk_text_document(text, file_meta)
            return []

        # PPTX (python-pptx 提取幻灯片文本)
        elif ext == ".pptx":
            try:
                from pptx import Presentation
                prs = Presentation(file_path)
                slides_text = []
                for slide in prs.slides:
                    texts = []
                    for shape in slide.shapes:
                        if shape.has_text_frame:
                            for para in shape.text_frame.paragraphs:
                                t = para.text.strip()
                                if t:
                                    texts.append(t)
                    if texts:
                        slides_text.append("\n".join(texts))
                text = "\n\n".join(slides_text)
                if text.strip():
                    return self.chunker.chunk_text_document(text, file_meta)
                return []
            except ImportError:
                print("   [WARN] python-pptx 未安装，无法解析 .pptx")
                return []
            except Exception as e:
                print(f"   [WARN] PPTX 解析失败: {e}")
                return []

        # PPT (旧版) — 尝试用 Windows COM 或 LibreOffice
        elif ext == ".ppt":
            text = self._parse_ppt_file(file_path)
            if text:
                return self.chunker.chunk_text_document(text, file_meta)
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

        # 方案6: raw binary UTF-16LE 兜底 (纯二进制提取, 零依赖)
        text = self._parse_doc_via_raw_binary(file_path)
        if text and text.strip():
            print(f"   [doc] raw binary 解析成功: {len(text)} 字符")
            return text

        print(f"   [warn] 所有 .doc 解析方案均失败: {os.path.basename(file_path)}")
        print(f"   [tip] 建议方案: (1) pip install pywin32 启用 Word COM 解析")
        print(f"         或 (2) 安装 LibreOffice")
        print(f"         或 (3) 用 Word 打开后另存为 .docx 格式")
        return ""

    def _parse_doc_via_raw_binary(self, file_path: str) -> str:
        """
        OLE2 .doc 纯二进制兜底提取 — 零外部依赖

        直接以 UTF-16LE 解码文件二进制内容，用正则提取中文文本段。
        适用于 OLE2 容器中文档内容以 Unicode 存储的常见情况。
        对复杂排版（表格/图片/公式）可能丢失部分内容，但绝大多数
        .doc 文件能够提取到 80%+ 的正文文本。
        """
        import re
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            # OLE2 .doc Word文档内文本以 UTF-16LE 编码存储
            text = data.decode('utf-16-le', errors='ignore')
            # 提取有意义的文本段: 4个字符以上, 中文占比 > 10%
            chunks = re.findall(
                r"[一-鿿a-zA-Z0-9\s.,;:!?()（）、。，；：！？''【】《》——…/%+\-]{4,}",
                text
            )
            # 过滤乱码: 中文占比太低的多半是字体名/样式名等垃圾
            clean = []
            for c in chunks:
                cn_count = sum(1 for ch in c if '一' <= ch <= '鿿')
                ratio = cn_count / max(len(c), 1)
                if ratio > 0.05 and len(c) > 6:
                    clean.append(c)
            return '\n'.join(clean)
        except Exception:
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

    # ===== WPS 文件解析 (.wps) =====

    def _parse_wps_file(self, file_path: str) -> str:
        """
        解析 .wps 文件 (金山 WPS 文字文档)

        WPS 有两代格式:
          - 新版 WPS (2010+): ZIP + XML 容器, 内部结构与 .docx 相同
          - 旧版 WPS (2005 及以前): OLE2 二进制容器, 类似旧 .doc

        按优先级尝试多种后端:
          1. python-docx (新版 .wps 本质是 .docx)
          2. zipfile 直接解压提取 XML 文本 (新版备选)
          3. LibreOffice headless 转换 (旧版+新版通用)
          4. olefile 原始提取 (旧版 OLE2)
          5. win32com WPS Office COM 自动化 (最可靠, 需安装 WPS Office)
          6. win32com MS Word COM 自动化 (MS Word 可能能打开)
        """
        abs_path = os.path.abspath(file_path)
        fname = os.path.basename(file_path)

        # 方案1: python-docx (新版 .wps = .docx)
        text = self._parse_wps_via_docx(file_path)
        if text and text.strip():
            print(f"   [wps] python-docx 解析成功: {len(text)} 字符")
            return text

        # 方案2: zipfile 直接提取 XML (新版 .wps)
        text = self._parse_wps_via_zip(file_path)
        if text and text.strip():
            print(f"   [wps] zipfile 解析成功: {len(text)} 字符")
            return text

        # 方案3: LibreOffice headless 转换 (通用)
        text = self._parse_doc_via_libreoffice(file_path)
        if text and text.strip():
            print(f"   [wps] LibreOffice 解析成功: {len(text)} 字符")
            return text

        # 方案4: olefile 原始提取 (旧版 .wps OLE2)
        text = self._parse_doc_via_olefile(file_path)
        if text and text.strip():
            print(f"   [wps] olefile 解析成功: {len(text)} 字符 (可能不完整)")
            return text

        # 方案5: WPS Office COM 自动化 (需安装 WPS Office)
        text = self._parse_wps_via_wps_com(file_path)
        if text and text.strip():
            print(f"   [wps] WPS COM 解析成功: {len(text)} 字符")
            return text

        # 方案6: MS Word COM (可能兼容某些 .wps)
        text = self._parse_doc_via_win32(file_path)
        if text and text.strip():
            print(f"   [wps] Word COM 解析成功: {len(text)} 字符")
            return text

        print(f"   [warn] 所有 .wps 解析方案均失败: {fname}")
        print(f"   [tip] 建议方案: (1) 安装 WPS Office 启用 COM 解析")
        print(f"         或 (2) 安装 LibreOffice")
        print(f"         或 (3) 用 WPS 打开后另存为 .docx 格式")
        return ""

    def _parse_wps_via_docx(self, file_path: str) -> str:
        """
        尝试用 python-docx 解析 .wps。
        新版 WPS 文件本质上是 ZIP + XML，与 .docx 结构相同。
        """
        try:
            import docx
            doc = docx.Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            if paragraphs:
                return "\n".join(paragraphs)
        except Exception:
            pass
        return ""

    def _parse_wps_via_zip(self, file_path: str) -> str:
        """
        尝试用 zipfile 直接解压 .wps 并提取 XML 中的文本。
        新版 WPS 本质是 ZIP 包，内含 word/document.xml 等。
        作为 python-docx 失败时的备选方案。
        """
        import zipfile
        from xml.etree import ElementTree as ET

        try:
            with zipfile.ZipFile(file_path, 'r') as z:
                # 检查是否为有效的 ZIP (新版 WPS 必备特征)
                if 'word/document.xml' not in z.namelist():
                    return ""

                # 提取主文档 XML
                xml_content = z.read('word/document.xml')

                # 从 XML 中提取所有文本节点
                root = ET.fromstring(xml_content)
                ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                texts = []
                for t in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
                    if t.text:
                        texts.append(t.text)
                return "\n".join(texts)
        except Exception:
            pass
        return ""

    def _parse_wps_via_wps_com(self, file_path: str) -> str:
        """
        通过 WPS Office COM 自动化提取文本。
        WPS Office 提供与 MS Word 兼容的 COM 接口 (ProgID: "WPS.Application"
        或 "KWPS.Application" 或 "ET.Application")。
        """
        abs_path = os.path.abspath(file_path)
        wps = None
        doc = None

        try:
            import pythoncom
            import win32com.client
        except ImportError:
            return ""

        # WPS Office 可能的 COM ProgID (按优先级)
        progids = [
            "WPS.Application",       # WPS Office 标准安装
            "KWPS.Application",      # Kingsoft WPS (旧版)
            "WPS.Document",          # 直接文档对象
            "Word.Application",      # MS Word (已在上层尝试)
        ]

        for progid in progids:
            if progid == "Word.Application":
                continue  # 已由上层 _parse_doc_via_win32 尝试

            try:
                pythoncom.CoInitialize()
                wps = win32com.client.Dispatch(progid)
                wps.Visible = False
                wps.DisplayAlerts = 0

                try:
                    doc = wps.Documents.Open(abs_path, ReadOnly=True, Visible=False)
                    text = doc.Content.Text

                    if text and text.strip():
                        return text
                except Exception:
                    pass
                finally:
                    if doc is not None:
                        try:
                            doc.Close(SaveChanges=False)
                        except Exception:
                            pass
                        doc = None
                    if wps is not None:
                        try:
                            wps.Quit()
                        except Exception:
                            pass
                        wps = None
            except Exception:
                pass
            finally:
                try:
                    pythoncom.CoUninitialize()
                except Exception:
                    pass

        return ""

    def _parse_ppt_file(self, file_path: str) -> str:
        """
        解析旧版 .ppt 文件 (PowerPoint 97-2003, OLE2 二进制)

        按优先级尝试:
          1. LibreOffice headless → txt (通用, 跨平台)
          2. win32com PowerPoint COM 自动化 (Windows + Office)
        """
        import subprocess
        import tempfile

        abs_path = os.path.abspath(file_path)

        # 方案1: LibreOffice headless 转 txt
        lo_paths = [
            "libreoffice", "soffice",
            "C:\\Program Files\\LibreOffice\\program\\soffice.exe",
            "C:\\Program Files (x86)\\LibreOffice\\program\\soffice.exe",
        ]
        for lo_exe in lo_paths:
            try:
                subprocess.run([lo_exe, "--version"], capture_output=True, timeout=5)
                break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                lo_exe = None
                continue

        if lo_exe:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    cmd = [lo_exe, "--headless", "--convert-to", "txt:Text",
                           "--outdir", tmpdir, abs_path]
                    result = subprocess.run(cmd, capture_output=True, timeout=120)
                    if result.returncode == 0:
                        for f in os.listdir(tmpdir):
                            if f.endswith(".txt"):
                                txt_path = os.path.join(tmpdir, f)
                                with open(txt_path, "r", encoding="utf-8", errors="ignore") as fp:
                                    text = fp.read()
                                if text.strip():
                                    return text
            except Exception:
                pass

        # 方案2: PowerPoint COM 自动化
        try:
            import pythoncom
            import win32com.client
        except ImportError:
            return ""

        ppt_app = None
        presentation = None
        try:
            pythoncom.CoInitialize()
            ppt_app = win32com.client.Dispatch("PowerPoint.Application")
            ppt_app.Visible = False
            ppt_app.DisplayAlerts = 0
            presentation = ppt_app.Presentations.Open(abs_path, ReadOnly=True)

            slides_text = []
            for slide in presentation.Slides:
                texts = []
                for shape in slide.Shapes:
                    if shape.HasTextFrame:
                        t = shape.TextFrame.TextRange.Text
                        if t and t.strip():
                            texts.append(t.strip())
                if texts:
                    slides_text.append("\n".join(texts))

            return "\n\n".join(slides_text)
        except Exception:
            return ""
        finally:
            if presentation is not None:
                try: presentation.Close()
                except Exception: pass
            if ppt_app is not None:
                try: ppt_app.Quit()
                except Exception: pass
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

    def _parse_ofd_file(self, file_path: str) -> str:
        """
        OFD 文件解析 — 带多级回退

        国产版式文档 OFD (GB/T 33190) 本质是 ZIP+XML 容器。
        优先级:
          1. 自定义提取器 (零依赖, 处理 GBK/UTF-8 混合编码 + 字体映射)
          2. ofdparser 库 (需安装: pip install ofdparser reportlab xmltodict)
        """
        # 方案1: 自定义提取器 (可靠, 无外部依赖)
        from ingestion.ofd_extractor import extract_ofd_text
        text = extract_ofd_text(file_path)
        if text and text.strip():
            return text

        # 方案2: ofdparser 库 (兜底, 处理特殊 OFD 变体)
        from ingestion.ofd_extractor import extract_ofd_text_via_ofdparser
        text = extract_ofd_text_via_ofdparser(file_path)
        if text and text.strip():
            return text

        return ""
