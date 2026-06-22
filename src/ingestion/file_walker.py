"""
文件遍历器 + SQLite 元数据提取
从知识库目录结构和文件名中自动提取元数据标签
"""

import os
import re
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import yaml


class FileWalker:
    """遍历知识库目录，提取每个文件的结构化元数据"""

    def __init__(self, config_path: str = None):
        from config import load_config
        self.config = load_config(config_path)

        self.kb_path = Path(self.config["paths"]["knowledge_base"])
        self.db_path = self.config["paths"]["metadata_db"]

        # 域关键词典
        self.domain_keywords = self.config.get("domain_keywords", {})
        # 发布层级正则
        self.publish_patterns = self.config.get("publish_level_patterns", {})
        # 电压等级正则
        self.voltage_patterns = self.config.get("voltage_patterns", {})
        # 文档编号正则
        self.doc_number_patterns = self.config.get("doc_number_patterns", {})
        # 专业类型正则
        self.discipline_patterns = self.config.get("discipline_patterns", {})
        # 设备类型正则
        self.equipment_patterns = self.config.get("equipment_patterns", {})
        # 图号正则
        self.drawing_code_pattern = self.config.get("drawing_code_pattern", "")
        self.drawing_discipline_map = self.config.get("drawing_discipline_map", {})

        self._init_db()

    def _get_db_connection(self):
        """获取 SQLite 连接，统一启用 WAL + busy_timeout 防锁"""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self):
        """初始化 SQLite 元数据库"""
        conn = self._get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT UNIQUE NOT NULL,
                full_path TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                extension TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                modified_time TEXT NOT NULL,

                -- 路径派生元数据
                domain TEXT,
                category TEXT,
                subcategory TEXT,
                path_depth INTEGER DEFAULT 0,
                path_year INTEGER,

                -- 文件名派生元数据
                doc_number TEXT,
                publish_level TEXT,
                voltage_level TEXT,
                discipline TEXT,
                equipment_type TEXT,
                year INTEGER,
                region TEXT DEFAULT '全国',
                drawing_code TEXT,

                -- 文件属性
                is_archive INTEGER DEFAULT 0,
                is_drawing INTEGER DEFAULT 0,
                file_format_group TEXT,

                -- 处理状态
                parsed INTEGER DEFAULT 0,
                parse_error TEXT,
                chunk_count INTEGER DEFAULT 0,

                created_at TEXT DEFAULT (datetime('now', 'localtime')),
                updated_at TEXT DEFAULT (datetime('now', 'localtime'))
            )
        """)

        # 索引
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_domain ON files(domain)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON files(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_extension ON files(extension)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doc_number ON files(doc_number)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_publish_level ON files(publish_level)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_voltage_level ON files(voltage_level)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_parsed ON files(parsed)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_domain_category ON files(domain, category)")

        conn.commit()
        conn.close()

    def compute_hash(self, filepath: Path) -> str:
        """计算文件 SHA256 哈希（用于去重和增量更新）"""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def extract_path_metadata(self, relative_path: str) -> dict:
        """
        从目录路径提取元数据
        例: 变电/标准规范/GB50060-2008.pdf
          → domain=变电, category=标准规范
        """
        parts = relative_path.replace("\\", "/").split("/")
        metadata = {
            "domain": None,
            "category": None,
            "subcategory": None,
            "path_depth": len(parts) - 1,
            "path_year": None,
        }

        if len(parts) >= 1:
            domain = parts[0]
            if domain in self.config["domains"]:
                metadata["domain"] = domain

        if len(parts) >= 2:
            metadata["category"] = parts[1]

        if len(parts) >= 3:
            # 排除文件名
            candidate = parts[2]
            # 不是文件名（没有扩展名或是目录）
            if "." not in candidate or candidate == parts[-1]:
                pass
            metadata["subcategory"] = candidate

        # 从路径段提取年份
        year_pattern = re.compile(r'^(19|20)\d{2}$')
        for part in parts:
            if year_pattern.match(part):
                metadata["path_year"] = int(part)
                break

        return metadata

    def extract_filename_metadata(self, filename: str, filepath: str) -> dict:
        """
        从文件名提取元数据
        使用正则匹配文档编号、发布层级、电压等级等
        """
        metadata = {
            "doc_number": None,
            "publish_level": None,
            "voltage_level": None,
            "discipline": None,
            "equipment_type": None,
            "year": None,
            "region": "全国",
            "drawing_code": None,
        }

        # 提取文档编号
        for key, pattern in self.doc_number_patterns.items():
            match = re.search(pattern, filename)
            if match:
                metadata["doc_number"] = match.group(0)
                break

        # 提取发布层级
        for level, patterns in self.publish_patterns.items():
            for pat in patterns:
                if re.search(pat, filename):
                    metadata["publish_level"] = self._normalize_publish_level(level)
                    break
            if metadata["publish_level"]:
                break

        # 提取电压等级
        for voltage, patterns in self.voltage_patterns.items():
            for pat in patterns:
                if re.search(pat, filename, re.IGNORECASE):
                    metadata["voltage_level"] = voltage
                    break
            if metadata["voltage_level"]:
                break

        # 提取专业类型
        for discipline, patterns in self.discipline_patterns.items():
            for pat in patterns:
                if re.search(pat, filename):
                    metadata["discipline"] = discipline
                    break
            if metadata["discipline"]:
                break

        # 提取设备类型
        for equipment, patterns in self.equipment_patterns.items():
            for pat in patterns:
                if re.search(pat, filename):
                    metadata["equipment_type"] = equipment
                    break
            if metadata["equipment_type"]:
                break

        # 提取图号 (如 110-A2-4-D0107-01)
        if self.drawing_code_pattern:
            match = re.search(self.drawing_code_pattern, filename)
            if match:
                metadata["drawing_code"] = match.group(0)
                disc_code = match.group("discipline_code")
                if disc_code in self.drawing_discipline_map:
                    if not metadata["discipline"]:
                        metadata["discipline"] = self.drawing_discipline_map[disc_code]
                if not metadata["voltage_level"]:
                    metadata["voltage_level"] = match.group("voltage") + "kV"

        # 提取年份
        year_match = re.search(r'((?:19|20)\d{2})', filename)
        if year_match:
            metadata["year"] = int(year_match.group(1))

        # 提取地域
        if re.search(r'福建|闽|福州|榕', filename):
            if re.search(r'福州|榕', filename):
                metadata["region"] = "福州"
            else:
                metadata["region"] = "福建"

        return metadata

    def _normalize_publish_level(self, raw: str) -> str:
        """规范化发布层级名称"""
        mapping = {
            "国标": "国标",
            "行标": "行标",
            "企标": "企标",
            "国网公司": "国网公司",
            "省公司_福建": "省公司",
            "地市公司_福州": "地市公司",
        }
        return mapping.get(raw, raw)

    def classify_file_format(self, extension: str, filepath: str) -> dict:
        """判断文件格式分组和是否为图纸/档案"""
        ext = extension.lower()
        is_drawing = 0
        is_archive = 0
        format_group = "other"

        if ext in (".dwg", ".dxf", ".dwl", ".dwl2"):
            is_drawing = 1
            format_group = "drawing"
        elif ext in (".zip", ".rar", ".7z"):
            is_archive = 1
            format_group = "archive"
        elif ext == ".pdf":
            # 从路径判断是否为图纸型 PDF
            path_lower = filepath.lower()
            if any(kw in path_lower for kw in ["设计图纸", "国网通用设计", "福州主接线", "模块化建设", "cad"]):
                is_drawing = 1
            format_group = "document"
        elif ext in (".doc", ".docx", ".ceb", ".ofd", ".wps"):
            format_group = "document"
        elif ext in (".xls", ".xlsx", ".et"):
            format_group = "spreadsheet"
        elif ext in (".ppt", ".pptx"):
            format_group = "presentation"
        elif ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"):
            format_group = "image"
        elif ext in (".chm", ".txt", ".md", ".html"):
            format_group = "text"

        return {
            "is_drawing": is_drawing,
            "is_archive": is_archive,
            "format_group": format_group,
        }

    def walk(self, batch_size: int = 1000, resume: bool = True) -> dict:
        """
        遍历知识库目录，提取并存储所有文件元数据
        返回统计信息
        """
        existing_hashes = set()
        if resume:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT file_hash FROM files")
            existing_hashes = {row[0] for row in cursor.fetchall()}
            conn.close()

        stats = {
            "total_files": 0,
            "new_files": 0,
            "skipped_files": 0,
            "error_files": 0,
            "by_domain": {},
            "by_extension": {},
            "by_format_group": {},
        }

        batch = []
        conn = self._get_db_connection()

        try:
            for filepath in self.kb_path.rglob("*"):
                if not filepath.is_file():
                    continue

                stats["total_files"] += 1

                try:
                    file_hash = self.compute_hash(filepath)
                    if file_hash in existing_hashes:
                        stats["skipped_files"] += 1
                        continue

                    relative_path = str(filepath.relative_to(self.kb_path))
                    stat = filepath.stat()

                    # 提取元数据
                    path_meta = self.extract_path_metadata(relative_path)
                    filename_meta = self.extract_filename_metadata(filepath.name, str(filepath))

                    # 合并域名（优先路径派生）
                    domain = path_meta["domain"]

                    # 扩展名分组
                    ext = filepath.suffix.lower()
                    format_info = self.classify_file_format(ext, relative_path)

                    row = (
                        file_hash,
                        str(filepath),
                        relative_path,
                        filepath.name,
                        ext,
                        stat.st_size,
                        datetime.fromtimestamp(stat.st_mtime).isoformat(),

                        domain,
                        path_meta["category"],
                        path_meta.get("subcategory"),
                        path_meta["path_depth"],
                        path_meta["path_year"],

                        filename_meta["doc_number"],
                        filename_meta["publish_level"],
                        filename_meta["voltage_level"],
                        filename_meta["discipline"],
                        filename_meta["equipment_type"],
                        filename_meta["year"],
                        filename_meta["region"],
                        filename_meta["drawing_code"],

                        format_info["is_archive"],
                        format_info["is_drawing"],
                        format_info["format_group"],
                    )

                    batch.append(row)
                    stats["new_files"] += 1

                    # 统计
                    stats["by_domain"][domain or "未分类"] = stats["by_domain"].get(domain or "未分类", 0) + 1
                    stats["by_extension"][ext] = stats["by_extension"].get(ext, 0) + 1
                    stats["by_format_group"][format_info["format_group"]] = \
                        stats["by_format_group"].get(format_info["format_group"], 0) + 1

                    # 批量写入
                    if len(batch) >= batch_size:
                        self._insert_batch(conn, batch)
                        batch = []

                    if stats["new_files"] % 5000 == 0:
                        print(f"  已处理新文件: {stats['new_files']} ...")

                except Exception as e:
                    stats["error_files"] += 1
                    if stats["error_files"] <= 10:
                        print(f"  ⚠ 错误 {filepath}: {e}")

            # 写入剩余批次
            if batch:
                self._insert_batch(conn, batch)

            conn.commit()

        finally:
            conn.close()

        return stats

    def _insert_batch(self, conn, batch: List[tuple]):
        """批量插入文件元数据"""
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT OR REPLACE INTO files (
                file_hash, full_path, relative_path, file_name, extension,
                size_bytes, modified_time,
                domain, category, subcategory, path_depth, path_year,
                doc_number, publish_level, voltage_level, discipline,
                equipment_type, year, region, drawing_code,
                is_archive, is_drawing, file_format_group
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch)
        conn.commit()


def build_file_index(config_path: str = None):
    """构建文件索引（入口函数）"""
    print("=" * 60)
    print("📁 榕能电力审图知识库 — 文件索引构建器")
    print("=" * 60)

    walker = FileWalker(config_path)
    print(f"知识库路径: {walker.kb_path}")
    print(f"元数据库: {walker.db_path}")
    print(f"\n开始遍历...")

    import time
    start = time.time()
    stats = walker.walk(batch_size=1000, resume=True)
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"📊 遍历统计 (耗时 {elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"  扫描文件总数: {stats['total_files']}")
    print(f"  新增/更新的文件: {stats['new_files']}")
    print(f"  已跳过(未变更): {stats['skipped_files']}")
    print(f"  错误文件: {stats['error_files']}")
    print(f"\n📂 按专业域分布:")
    for domain, count in sorted(stats['by_domain'].items(), key=lambda x: -x[1]):
        print(f"    {domain}: {count} 个文件")
    print(f"\n📄 按文件格式分布 (Top 15):")
    for ext, count in sorted(stats['by_extension'].items(), key=lambda x: -x[1])[:15]:
        print(f"    {ext}: {count}")
    print(f"\n📦 按格式分组:")
    for group, count in sorted(stats['by_format_group'].items(), key=lambda x: -x[1]):
        print(f"    {group}: {count}")


if __name__ == "__main__":
    build_file_index()
