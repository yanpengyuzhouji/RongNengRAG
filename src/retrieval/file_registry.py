"""
文件注册表查询模块 — 从 SQLite 注册表中查找文件名匹配
用于检测用户查询中是否引用了特定文件，并获取完整文档内容注入 prompt
"""

import os
import re
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class FileRegistryEntry:
    """文件注册表条目"""
    file_hash: str
    original_path: str
    stored_path: str
    file_name: str
    file_size: int
    file_type: str
    status: str
    chunks_count: int
    chars_count: int
    domain: str
    category: str
    doc_number: str
    created_at: str
    updated_at: str

    @property
    def name_without_ext(self) -> str:
        """返回不带扩展名的文件名"""
        return os.path.splitext(self.file_name)[0]

    @property
    def ext(self) -> str:
        """返回扩展名（含点号）"""
        return os.path.splitext(self.file_name)[1].lower()


@dataclass
class FileMatchResult:
    """文件名匹配结果"""
    entry: FileRegistryEntry
    match_type: str       # "exact" | "without_ext" | "partial" | "extension_pattern"
    match_score: float    # 0~1
    matched_text: str     # 在 query 中匹配到的文本片段


class FileRegistry:
    """
    文件注册表查询器

    从 SQLite file_registry 表中查询已入库的文件，
    支持按文件名片段搜索，以及检测用户 query 中的文件名引用。

    用法:
        registry = FileRegistry()
        # 查找文件名匹配
        matches = registry.detect_files_in_query("GB 50060-2008 的防火间距要求")
        # 获取所有注册文件名
        names = registry.get_all_filenames()
    """

    # query 中识别文件名的模式
    FILENAME_PATTERNS = [
        # 带引号或书名号
        re.compile(r'["""]([^"""]+?\.[a-zA-Z0-9]{2,5})["""]'),
        re.compile(r'《([^》]+?\.[a-zA-Z0-9]{2,5})》'),
        # 明确的文件名（含扩展名）
        re.compile(r'([\w一-鿿\-\s()（）]+\.(?:pdf|doc|docx|xls|xlsx|ppt|pptx|txt|ofd|ceb|dwg))',
                   re.IGNORECASE),
        # 文件名不含扩展名但明确提及"文件"
        re.compile(r'文件[：:]\s*([^\s，。,\.]+)'),
        re.compile(r'文件\s*[《"]([^》"]+)[》"]'),
    ]

    def __init__(self, config_path: str = None):
        from config import load_config
        self.config = load_config(config_path)
        self.db_path = self.config["paths"]["metadata_db"]

        # 缓存
        self._filename_cache: Optional[List[FileRegistryEntry]] = None
        self._cache_ttl: float = 60.0  # 60秒缓存
        self._cache_time: float = 0.0

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_all_entries(self, status: str = "completed") -> List[FileRegistryEntry]:
        """获取所有已入库文件的注册表条目"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM file_registry WHERE status = ? ORDER BY file_name",
            (status,)
        )
        rows = cursor.fetchall()
        conn.close()

        entries = []
        for row in rows:
            d = dict(row)
            entries.append(FileRegistryEntry(
                file_hash=d.get("file_hash", ""),
                original_path=d.get("original_path", ""),
                stored_path=d.get("stored_path", ""),
                file_name=d.get("file_name", ""),
                file_size=d.get("file_size", 0),
                file_type=d.get("file_type", ""),
                status=d.get("status", ""),
                chunks_count=d.get("chunks_count", 0),
                chars_count=d.get("chars_count", 0),
                domain=d.get("domain", ""),
                category=d.get("category", ""),
                doc_number=d.get("doc_number", ""),
                created_at=d.get("created_at", ""),
                updated_at=d.get("updated_at", ""),
            ))
        return entries

    def get_cached_entries(self) -> List[FileRegistryEntry]:
        """获取缓存的条目列表，过期自动刷新"""
        import time
        now = time.time()
        if (self._filename_cache is None or
                now - self._cache_time > self._cache_ttl):
            self._filename_cache = self.get_all_entries()
            self._cache_time = now
        return self._filename_cache

    def search_by_filename(self, fragment: str,
                           exact: bool = False) -> List[FileRegistryEntry]:
        """
        按文件名片段搜索

        Args:
            fragment: 文件名片段
            exact: True=精确匹配文件名, False=模糊匹配

        Returns:
            匹配的文件条目列表
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        if exact:
            cursor.execute(
                "SELECT * FROM file_registry WHERE status='completed' AND file_name = ?",
                (fragment,)
            )
        else:
            cursor.execute(
                "SELECT * FROM file_registry WHERE status='completed' AND file_name LIKE ?",
                (f"%{fragment}%",)
            )

        rows = cursor.fetchall()
        conn.close()

        entries = []
        for row in rows:
            d = dict(row)
            entries.append(FileRegistryEntry(
                file_hash=d.get("file_hash", ""),
                original_path=d.get("original_path", ""),
                stored_path=d.get("stored_path", ""),
                file_name=d.get("file_name", ""),
                file_size=d.get("file_size", 0),
                file_type=d.get("file_type", ""),
                status=d.get("status", ""),
                chunks_count=d.get("chunks_count", 0),
                chars_count=d.get("chars_count", 0),
                domain=d.get("domain", ""),
                category=d.get("category", ""),
                doc_number=d.get("doc_number", ""),
                created_at=d.get("created_at", ""),
                updated_at=d.get("updated_at", ""),
            ))
        return entries

    def detect_files_in_query(self, query: str,
                              min_score: float = 0.4) -> List[FileMatchResult]:
        """
        检测用户 query 中是否引用了已注册文件

        匹配策略（按优先级）:
        1. 正则提取 query 中的文件名模式
        2. 在注册表中查找匹配
        3. 反向：遍历注册表文件名，检查是否出现在 query 中

        Args:
            query: 用户查询文本
            min_score: 最低匹配分数阈值

        Returns:
            匹配结果列表，按分数降序排列
        """
        results: Dict[str, FileMatchResult] = {}  # key=file_hash, 去重

        # 策略1: 正则提取 query 中可能的文件名
        extracted_names = self._extract_filename_candidates(query)

        if extracted_names:
            entries = self.get_cached_entries()
            for candidate_name in extracted_names:
                clean = candidate_name.strip().lower()
                for entry in entries:
                    if entry.file_hash in results:
                        continue

                    entry_name_lower = entry.file_name.lower()
                    entry_name_noext = entry.name_without_ext.lower()

                    # 精确匹配（含扩展名）
                    if clean == entry_name_lower:
                        results[entry.file_hash] = FileMatchResult(
                            entry=entry,
                            match_type="exact",
                            match_score=1.0,
                            matched_text=candidate_name,
                        )
                    # 不含扩展名匹配
                    elif clean == entry_name_noext:
                        results[entry.file_hash] = FileMatchResult(
                            entry=entry,
                            match_type="without_ext",
                            match_score=0.95,
                            matched_text=candidate_name,
                        )
                    # 候选名包含完整文件名
                    elif len(clean) >= 6 and entry_name_lower in clean:
                        results[entry.file_hash] = FileMatchResult(
                            entry=entry,
                            match_type="partial",
                            match_score=0.85,
                            matched_text=candidate_name,
                        )
                    # 文件名包含候选名
                    elif len(clean) >= 4 and clean in entry_name_lower:
                        results[entry.file_hash] = FileMatchResult(
                            entry=entry,
                            match_type="partial",
                            match_score=0.75,
                            matched_text=candidate_name,
                        )

        # 策略2: 双向匹配 — query ↔ 文件名互相包含
        # (针对没有扩展名/引号但文件名关键词在 query 中的情况)
        entries = self.get_cached_entries()
        query_lower = query.lower()
        # 提取 query 中有意义的关键词片段（用于匹配文件名）
        query_phrases = self._extract_query_phrases(query)

        for entry in entries:
            if entry.file_hash in results:
                continue

            name_lower = entry.file_name.lower()
            name_noext = entry.name_without_ext.lower()

            # 2a: 文件名包含在 query 中（原有的反向匹配）
            if len(name_noext) >= 6 and name_noext in query_lower:
                results[entry.file_hash] = FileMatchResult(
                    entry=entry,
                    match_type="partial",
                    match_score=0.7,
                    matched_text=entry.file_name,
                )
            elif len(name_lower) >= 8 and name_lower in query_lower:
                results[entry.file_hash] = FileMatchResult(
                    entry=entry,
                    match_type="partial",
                    match_score=0.65,
                    matched_text=entry.file_name,
                )

            # 2b: query 关键词包含在文件名中（新增的正向匹配）
            # 解决 "会议材料之一" 匹配 "01会议材料之一2022年..." 的问题
            else:
                for phrase, phrase_score in query_phrases:
                    if phrase in name_noext:
                        results[entry.file_hash] = FileMatchResult(
                            entry=entry,
                            match_type="partial",
                            match_score=phrase_score,
                            matched_text=phrase,
                        )
                        break

        # 策略3: 文档编号匹配 (如 "闽电发展〔2015〕241号")
        doc_patterns = [
            r'[闽榕国]\S*〔\d{4}〕\d+号',
            r'[A-Z]{2,}[/T]?\s*\d+[\.\-]\d+',
            r'Q/GDW\s*\d+[\.\-]\d+',
        ]
        for pattern in doc_patterns:
            for match in re.finditer(pattern, query):
                doc_num = match.group(0)
                for entry in entries:
                    if entry.file_hash in results:
                        continue
                    if entry.doc_number and doc_num in entry.doc_number:
                        results[entry.file_hash] = FileMatchResult(
                            entry=entry,
                            match_type="partial",
                            match_score=0.8,
                            matched_text=doc_num,
                        )

        # 策略4: "会议材料之X" / "会议材料X" 模式 — 支持五种写法:
        #   "01会议材料之一" (完整，含数字前缀+之)
        #   "会议材料之一"   (无数字前缀，有之)
        #   "01会议材料一"   (有数字前缀，无之)
        #   "会议材料一"     (无数字前缀，无之 — 用户最常见的自然写法)
        #   "之一" / "之三"  (仅序号，上下文含材料/会议)
        conf_material = None
        # 先尝试匹配有"之"的写法
        for cm_pat in [
            r'(\d{2,4})\s*会议材料之([一二三四五六七八九十\d]+)',   # 01会议材料之一
            r'会议材料之([一二三四五六七八九十\d]+)',                # 会议材料之一
        ]:
            conf_material = re.search(cm_pat, query)
            if conf_material:
                break

        # 再尝试匹配无"之"的写法 (最常见自然语言: "会议材料一" "会议材料三")
        if not conf_material:
            for cm_pat in [
                r'(\d{2,4})\s*会议材料([一二三四五六七八九十\d]+)',  # 01会议材料一
                r'会议材料([一二三四五六七八九十\d]+)',               # 会议材料一
            ]:
                conf_material = re.search(cm_pat, query)
                if conf_material:
                    break

        # 更宽松: "之X" 但 query 中有 "材料" 或 "会议"
        if not conf_material:
            if '材料' in query or '会议' in query:
                conf_material = re.search(r'之([一二三四五六七八九十\d]+)', query)
        # 最后: "材料一" "材料三" (仅序号，上下文含会议/材料)
        if not conf_material:
            if '会议' in query or '材料' in query:
                conf_material = re.search(r'材料([一二三四五六七八九十\d]+)', query)

        if conf_material:
            num = conf_material.group(1) if conf_material.lastindex >= 1 else None
            seq = conf_material.group(2) if conf_material.lastindex >= 2 else conf_material.group(1)

            seq_digit = self.CN_TO_DIGIT.get(seq, seq)
            if seq_digit.isdigit():
                seq_digit = seq_digit.zfill(2)  # 对齐 "01" 格式

            for entry in entries:
                if entry.file_hash in results:
                    continue
                fname = entry.file_name
                if '会议材料' not in fname and '材料' not in fname:
                    continue

                # 匹配序号: 文件名中的 "之X" 序号与 query 一致
                seq_in_name = re.search(r'之([一二三四五六七八九十\d]+)', fname)
                if seq_in_name:
                    name_seq = seq_in_name.group(1)
                    name_seq_digit = self.CN_TO_DIGIT.get(name_seq, name_seq)
                    if name_seq_digit.isdigit():
                        name_seq_digit = name_seq_digit.zfill(2)

                    if name_seq_digit == seq_digit:
                        match_score = 0.95
                    elif name_seq_digit.lstrip('0') == seq_digit.lstrip('0'):
                        match_score = 0.95
                    else:
                        continue  # 序号不匹配，跳过
                else:
                    # 文件名没有 "之X" 模式但有 "材料"
                    match_score = 0.6

                # 如果有数字前缀(如 01)，加分验证
                if num and num.isdigit():
                    num_prefix = num.zfill(2)
                    if fname.startswith(num_prefix):
                        match_score = 1.0  # 完整匹配
                    elif num in fname:
                        match_score = max(match_score, 0.85)

                results[entry.file_hash] = FileMatchResult(
                    entry=entry,
                    match_type="partial",
                    match_score=match_score,
                    matched_text=conf_material.group(0),
                )

        # 过滤低分结果并排序
        filtered = [r for r in results.values() if r.match_score >= min_score]
        filtered.sort(key=lambda x: -x.match_score)
        return filtered

    def _extract_filename_candidates(self, query: str) -> List[str]:
        """
        从 query 中提取可能的文件名候选

        Returns:
            候选文件名列表
        """
        candidates = []

        # 书名号中的内容
        for m in re.finditer(r'《([^》]+)》', query):
            content = m.group(1).strip()
            if len(content) >= 3:
                candidates.append(content)

        # 引号中的内容
        for m in re.finditer(r'["""]([^"""]+?)["""]', query):
            content = m.group(1).strip()
            if len(content) >= 3:
                candidates.append(content)

        # 明确的文件名模式（含扩展名）
        for m in re.finditer(
            r'([\w一-鿿\-\s()（）]+\.(?:pdf|doc|docx|xls|xlsx|ppt|pptx|txt|ofd|ceb|dwg))',
            query, re.IGNORECASE
        ):
            candidates.append(m.group(1).strip())

        # "XX文件" 模式
        for m in re.finditer(r'([^\s，。,\.]{2,30})\s*文件', query):
            candidates.append(m.group(1).strip())

        return candidates

    # 中文数字 → 阿拉伯数字映射（模块级复用）
    CN_TO_DIGIT = {
        '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
        '六': '6', '七': '7', '八': '8', '九': '9', '十': '10',
        '十一': '11', '十二': '12', '十三': '13', '十四': '14',
        '十五': '15', '十六': '16', '十七': '17', '十八': '18',
        '十九': '19', '二十': '20',
    }

    def _extract_query_phrases(self, query: str) -> List[tuple]:
        """
        从 query 中提取关键词片段，用于与文件名做双向匹配。
        返回 [(phrase, score), ...] 列表，score 为匹配置信度。

        解决 "会议材料之一" 匹配 "01会议材料之一2022年..." 的问题：
        - 原逻辑只检查 长文件名 in 短query → 永远失败
        - 新逻辑同时检查 query片段 in 文件名 → 可以匹配

        策略:
        - 提取长度 >= 4 的中文/英文连续片段
        - 生成数字变体: "之一" → ["之一", "01", "1"]
        - 片段越长越具体，score 越高
        """
        import re as _re
        phrases = []

        # 1. 提取 "会议材料之X" / "会议材料X" 模式（高优先级）
        # 有"之"的写法
        m = _re.search(r'会议材料之([一二三四五六七八九十\d]+)', query)
        if not m:
            # 无"之"的写法: "会议材料一" "会议材料三" (最常见的自然语言)
            m = _re.search(r'会议材料([一二三四五六七八九十\d]+)', query)
        if not m:
            # 数字前缀+材料: "01会议材料一"
            m = _re.search(r'(\d{2,4})\s*会议材料([一二三四五六七八九十\d]+)', query)
        if m:
            # 提取序号
            if m.lastindex >= 2:
                seq = m.group(2)
                num_prefix = m.group(1) if m.lastindex >= 2 else None
            else:
                seq = m.group(1)
                num_prefix = None
            seq_digit = self.CN_TO_DIGIT.get(seq, seq)
            # 原始匹配文本
            phrases.append((m.group(0), 0.88))
            # 数字前缀变体: "会议材料一" → "01会议材料一"
            if seq_digit.isdigit():
                prefixed = seq_digit.zfill(2) + m.group(0)
                phrases.append((prefixed, 0.75))
            # 带"之"的变体: "会议材料一" → "会议材料之一"
            phrases.append((f'会议材料之{seq}', 0.85))
            # "材料之X" 片段
            phrases.append(('材料之' + seq, 0.80))
            phrases.append(('材料' + seq, 0.70))

        # 2. 提取 "之X" / "材料X" 模式（query 含 "材料"/"会议" 时）
        if '材料' in query or '会议' in query:
            for cm in _re.finditer(r'之([一二三四五六七八九十\d]+)', query):
                seq = cm.group(1)
                seq_digit = self.CN_TO_DIGIT.get(seq, seq)
                phrases.append((cm.group(0), 0.65))
                if seq_digit.isdigit():
                    phrases.append(('材料之' + seq, 0.68))
            # 无"之"的变体: "材料一" "材料三"
            for cm in _re.finditer(r'材料([一二三四五六七八九十\d]+)', query):
                seq = cm.group(1)
                seq_digit = self.CN_TO_DIGIT.get(seq, seq)
                phrases.append((cm.group(0), 0.62))
                phrases.append(('材料之' + seq, 0.65))

        # 3. 提取中文连续片段（>= 4 字）
        for cm in _re.finditer(r'[一-鿿\d]{4,}', query):
            phrase = cm.group(0)
            score = min(0.7, 0.45 + len(phrase) * 0.02)
            phrases.append((phrase, score))

        # 4. 数字前缀 + 简短关键词组合（如 "01会议"、"03材料"）
        for cm in _re.finditer(r'(\d{2})\s*(会议|材料|设计|电力|标准)', query):
            combined = cm.group(1) + cm.group(2)
            phrases.append((combined, 0.72))

        # 去重并按 score 降序排列
        seen = set()
        deduped = []
        for p, s in phrases:
            k = p.lower()
            if k not in seen:
                seen.add(k)
                deduped.append((p, s))
        deduped.sort(key=lambda x: -x[1])
        return deduped

    def get_entry_by_filename(self, filename: str) -> Optional[FileRegistryEntry]:
        """按精确文件名查找"""
        results = self.search_by_filename(filename, exact=True)
        return results[0] if results else None

    def get_entry_by_hash(self, file_hash: str) -> Optional[FileRegistryEntry]:
        """按文件哈希查找"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM file_registry WHERE file_hash = ?", (file_hash,))
        row = cursor.fetchone()
        conn.close()
        if row:
            d = dict(row)
            return FileRegistryEntry(
                file_hash=d.get("file_hash", ""),
                original_path=d.get("original_path", ""),
                stored_path=d.get("stored_path", ""),
                file_name=d.get("file_name", ""),
                file_size=d.get("file_size", 0),
                file_type=d.get("file_type", ""),
                status=d.get("status", ""),
                chunks_count=d.get("chunks_count", 0),
                chars_count=d.get("chars_count", 0),
                domain=d.get("domain", ""),
                category=d.get("category", ""),
                doc_number=d.get("doc_number", ""),
                created_at=d.get("created_at", ""),
                updated_at=d.get("updated_at", ""),
            )
        return None

    def get_file_count(self) -> int:
        """获取已入库文件总数"""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM file_registry WHERE status='completed'")
        count = cursor.fetchone()[0]
        conn.close()
        return count
