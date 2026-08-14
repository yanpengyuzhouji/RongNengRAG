"""
检索编排器 — 三阶段检索管道的顶层协调器
阶段0: 查询分析
阶段1: 粗召回 (混合搜索)
阶段2: 精排 (交叉编码器 + 元数据)

增强: 文件注册表识别 — 若 query 中含有文件名，则引入完整文档注入 prompt
"""

import sys
import os
import time
import json
import re
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieval.query_analyzer import QueryAnalyzer, AnalyzedQuery
from retrieval.reranker import Reranker
from retrieval.file_registry import (
    FileRegistry, FileRegistryEntry, FileMatchResult
)
from ingestion.embedder import Embedder
from ingestion.milvus_store import MilvusStore
from retrieval.cross_domain import apply_domain_override
from retrieval.context_selection import select_context_chunks
from retrieval.series import extract_series_key
from ingestion.document_editor import layout_page_texts
from retrieval.bm25_index import BM25Index


@dataclass
class RetrievalResult:
    """单条检索结果"""
    chunk_id: str
    text: str
    score: float
    confidence: float = 0.0
    domain: str = ""
    category: str = ""
    file_path: str = ""
    doc_number: str = ""
    voltage_level: str = ""
    publish_level: str = ""
    discipline: str = ""
    equipment_type: str = ""
    year: int = 0
    region: str = ""
    page_num: int = 0
    is_drawing: bool = False

    def to_context_str(self) -> str:
        """格式化为 LLM 上下文，文件名最前，防止 LLM 被内容中的数字误导"""
        import os
        meta_parts = []
        # 文件名放最前面，用 >>> 强化标识
        if self.file_path:
            fname = os.path.basename(self.file_path)
            meta_parts.append(f"文件: {fname}")
        if self.doc_number:
            meta_parts.append(f"编号: {self.doc_number}")
        if self.domain:
            meta_parts.append(f"域: {self.domain}")
        if self.category:
            meta_parts.append(f"类目: {self.category}")
        if self.voltage_level:
            meta_parts.append(f"电压: {self.voltage_level}")
        if self.publish_level:
            meta_parts.append(f"发布: {self.publish_level}")
        if self.page_num:
            meta_parts.append(f"页码: {self.page_num}")

        header = f"【{' | '.join(meta_parts)}】" if meta_parts else ""
        return f"{header}\n{self.text}"


@dataclass
class SearchResponse:
    """检索响应"""
    query: str
    query_type: str
    domain: Optional[str]
    results: List[RetrievalResult]
    total_candidates: int
    elapsed_ms: float
    filter_applied: Optional[str] = None
    expanded_terms: List[str] = field(default_factory=list)
    coarse_results: Optional[List[RetrievalResult]] = None  # 精排前候选快照（用于评估）


@dataclass
class StatisticalResult:
    """统计查询结果 — 全量元数据聚合"""
    query_type: str  # "count", "list", "distribution"
    total_chunks: int  # 匹配到的 chunk 总数
    unique_files: int  # 去重文件数
    by_domain: Dict[str, int] = field(default_factory=dict)
    by_category: Dict[str, int] = field(default_factory=dict)
    by_voltage_level: Dict[str, int] = field(default_factory=dict)
    by_publish_level: Dict[str, int] = field(default_factory=dict)
    by_discipline: Dict[str, int] = field(default_factory=dict)
    by_year: Dict[str, int] = field(default_factory=dict)
    file_list: List[Dict[str, str]] = field(default_factory=list)
    formatted_table: str = ""


class Retriever:
    """三阶段检索编排器"""

    def __init__(self, config_path: str = None):
        from config import load_config
        self.config = load_config(config_path)

        self.config_path = config_path
        self.analyzer = QueryAnalyzer(config_path)
        self.embedder = Embedder(config_path)
        self.store = MilvusStore(config_path)
        self.reranker = Reranker(config_path)
        self.file_registry = FileRegistry(config_path)
        self._layout_text_cache = {}
        self._bm25_index = None
        self._bm25_signature = None

        self.retrieval_config = self.config["retrieval"]

    def _ensure_bm25_index(self) -> Optional[BM25Index]:
        """Build BM25 from the currently published Milvus generation."""
        if not self.retrieval_config.get("bm25_enabled", True):
            return None
        try:
            target = self.store._active_target()
            stats = self.store.get_collection_stats()
            row_count = int(stats.get("count", 0))
            signature = (target, row_count)
            if self._bm25_index is not None and self._bm25_signature == signature:
                return self._bm25_index
            rows = self.store.query_rows_for_bm25()
            self._bm25_index = BM25Index(rows)
            # Use the observed row count as well as Milvus stats: a staged
            # generation may be visible before its stats are refreshed.
            self._bm25_signature = (target, len(rows))
            print(f"[bm25] 索引就绪: {len(rows)} 个 chunks, avgdl={self._bm25_index.avgdl:.1f}")
            return self._bm25_index
        except Exception as exc:
            # BM25 is an additional recall branch.  A Milvus/API deployment
            # must remain usable if a very large filtered snapshot is not
            # available on a particular Milvus Lite version.
            print(f"[bm25][WARN] 构建失败，回退向量检索: {exc}")
            return None

    def _merge_bm25_candidates(
        self, query: str, candidates: List[dict], filter_expr: Optional[str], limit: int
    ) -> List[dict]:
        """Fuse Milvus dense+BGE-M3 results with corpus-aware BM25 scores."""
        index = self._ensure_bm25_index()
        if index is None or index.empty:
            return candidates

        allowed_ids = None
        if filter_expr:
            try:
                allowed_ids = {
                    str(row.get("chunk_id"))
                    for row in self.store.query_rows_for_bm25(filter_expr)
                    if row.get("chunk_id")
                }
            except Exception as exc:
                print(f"[bm25][WARN] 过滤候选读取失败，忽略 BM25 过滤: {exc}")

        bm25_candidates = index.search(query, limit=max(limit, 1), allowed_ids=allowed_ids)
        if not bm25_candidates:
            return candidates

        vector_by_id = {}
        for item in candidates or []:
            entity = item.get("entity", item) if isinstance(item, dict) else {}
            chunk_id = str(entity.get("chunk_id") or item.get("id") or "")
            if chunk_id:
                vector_by_id[chunk_id] = item

        bm25_by_id = {
            str(item.get("entity", {}).get("chunk_id") or item.get("id")): item
            for item in bm25_candidates
        }
        all_ids = list(dict.fromkeys([*vector_by_id, *bm25_by_id]))
        max_bm25 = max((float(item.get("_bm25_score", 0.0)) for item in bm25_candidates), default=0.0)
        bm25_weight = min(0.8, max(0.0, float(self.retrieval_config.get("bm25_weight", 0.25))))
        vector_weight = 1.0 - bm25_weight
        merged = []
        for chunk_id in all_ids:
            vector_item = vector_by_id.get(chunk_id)
            bm25_item = bm25_by_id.get(chunk_id)
            item = dict(vector_item or bm25_item or {})
            entity = dict((vector_item or bm25_item).get("entity", {}))
            if bm25_item:
                # BM25 rows contain the complete scalar metadata snapshot.
                entity.update(bm25_item.get("entity", {}))
            item["entity"] = entity
            vector_score = float((vector_item or {}).get("distance", 0.0) or 0.0)
            vector_score = min(1.0, max(0.0, vector_score))
            bm25_score = float((bm25_item or {}).get("_bm25_score", 0.0) or 0.0)
            bm25_norm = bm25_score / max_bm25 if max_bm25 > 0 else 0.0
            # A lexical-only hit must remain eligible when the ANN branch did
            # not return the same chunk; otherwise the BM25 branch could never
            # recover an exact standard number missed by ANN.
            fused = (
                bm25_norm if not vector_item
                else vector_weight * vector_score + bm25_weight * bm25_norm
            )
            item["distance"] = float(fused)
            item["_vector_score"] = vector_score
            item["_bm25_score"] = bm25_score
            item["_retrieval_sources"] = (
                "vector+bm25" if vector_item else "bm25"
            )
            merged.append(item)
        merged.sort(key=lambda item: item.get("distance", 0.0), reverse=True)
        return merged[:limit]

    @staticmethod
    def _consistency_text(value: str) -> str:
        """Normalize layout/chunk text for stale-index consistency checks."""
        plain = re.sub(r"<[^>]+>", " ", str(value or ""))
        return re.sub(r"\s+", "", plain)

    def _current_layout_pages(self, file_hash: str):
        """Load the latest persisted layout text for one file, if available."""
        if not file_hash or len(file_hash) < 64:
            return None
        cache_dir = Path(self.config["paths"].get("parsed_cache", "data/parsed_cache"))
        candidates = [
            cache_dir / f"{file_hash}.layout.edited.json",
            cache_dir / f"{file_hash}.layout.json",
        ]
        for path in candidates:
            try:
                stat = path.stat()
            except OSError:
                continue
            stamp = (stat.st_mtime_ns, stat.st_size)
            cached = self._layout_text_cache.get(file_hash)
            if cached and cached[0] == str(path) and cached[1] == stamp:
                return cached[2]
            try:
                layout = json.loads(path.read_text(encoding="utf-8"))
                pages = {
                    page_num: text
                    for page_num, text in layout_page_texts(layout)
                    if str(text or "").strip()
                }
            except (OSError, ValueError, TypeError):
                continue
            self._layout_text_cache[file_hash] = (str(path), stamp, pages)
            return pages
        return None

    @staticmethod
    def _result_file_hash(item: dict) -> str:
        entity = item.get("entity", item) if isinstance(item, dict) else {}
        chunk_id = str(
            entity.get("chunk_id")
            or (item.get("chunk_id") if isinstance(item, dict) else "")
            or (item.get("id") if isinstance(item, dict) else "")
            or ""
        )
        prefix = chunk_id[:64]
        return prefix if re.fullmatch(r"[0-9a-f]{64}", prefix) else ""

    def _remove_stale_layout_results(self, candidates: list) -> list:
        """Reject chunks that are not present in the current edited layout."""
        filtered = []
        for item in candidates or []:
            entity = item.get("entity", item) if isinstance(item, dict) else {}
            file_hash = self._result_file_hash(item)
            pages = self._current_layout_pages(file_hash) if file_hash else None
            if pages is not None:
                candidate_text = self._consistency_text(entity.get("text", ""))
                try:
                    page_num = int(entity.get("page_num") or 0)
                except (TypeError, ValueError):
                    page_num = 0
                current_text = self._consistency_text(
                    pages.get(page_num, "") if page_num else "".join(pages.values())
                )
                if candidate_text and candidate_text not in current_text:
                    continue
            filtered.append(item)
        return filtered

    def search(self, query: str, top_k: int = None,
               domain_filter: str = None,
               return_coarse_results: bool = False) -> SearchResponse:
        """
        执行三阶段检索

        Args:
            query: 用户自然语言查询
            top_k: 返回结果数
            domain_filter: 手动指定域过滤（覆盖自动分析）

        Returns:
            SearchResponse
        """
        t_start = time.time()

        if top_k is None:
            top_k = self.retrieval_config["fine_top_k"]

        # ===== 阶段0: 查询分析 =====
        aq = self.analyzer.analyze(query)

        # 手动域过滤覆盖
        apply_domain_override(aq, domain_filter, self.analyzer._build_filter_expr)

        # ===== 阶段1: 混合搜索粗召回 =====
        # 对查询进行嵌入
        search_query = aq.expanded_query or query
        dense_vec, sparse_vec = self.embedder.encode_query(search_query)

        coarse_k = self.retrieval_config["coarse_top_k"]

        candidates = self.store.hybrid_search(
            dense_vector=dense_vec,
            sparse_vector=sparse_vec,
            filter_expr=aq.filter_expr,
            limit=coarse_k,
            rrf_k=self.retrieval_config["rrf_k"],
            dense_weight=self.retrieval_config["dense_weight"],
            sparse_weight=self.retrieval_config["sparse_weight"],
        )
        candidates = self._remove_stale_layout_results(candidates)
        # Milvus branch = Qwen dense + BGE-M3 learned sparse weights.
        # Add a corpus-aware BM25 branch for exact standards/numbers/keywords.
        candidates = self._merge_bm25_candidates(
            query=query,
            candidates=candidates,
            filter_expr=aq.filter_expr,
            limit=coarse_k,
        )

        # ===== 快照粗排候选（用于评估） =====
        coarse_snapshot = None
        if return_coarse_results:
            coarse_snapshot = []
            for item in candidates:
                entity = item.get("entity", item)
                raw_score = item.get("_rerank_score", item.get("distance", 0.0))
                confidence = round(max(0.0, min(1.0, float(raw_score))), 4)
                coarse_snapshot.append(RetrievalResult(
                    chunk_id=entity.get("chunk_id", ""),
                    text=entity.get("text", ""),
                    score=item.get("distance", 0.0) if "distance" in item else 0.0,
                    confidence=confidence,
                    domain=entity.get("domain", ""),
                    category=entity.get("category", ""),
                    file_path=entity.get("file_path", ""),
                    doc_number=entity.get("doc_number", ""),
                    voltage_level=entity.get("voltage_level", ""),
                    publish_level=entity.get("publish_level", ""),
                    discipline=entity.get("discipline", ""),
                    equipment_type=entity.get("equipment_type", ""),
                    year=entity.get("year", 0),
                    region=entity.get("region", ""),
                    page_num=entity.get("page_num", 0),
                    is_drawing=entity.get("is_drawing", False),
                ))
            coarse_snapshot.sort(key=lambda x: x.score, reverse=True)

        # ===== 阶段2: 交叉编码器精排 =====
        if len(candidates) <= top_k:
            ranked = candidates
            reranker_used = False
        else:
            try:
                ranked = self.reranker.rerank(
                    query=query,
                    candidates=candidates,
                    analyzed_query=aq,
                    top_k=top_k,
                )
                reranker_used = True
            except Exception as e:
                print(f"[warn] 重排序失败，回退元数据排序: {e}")
                ranked = self.reranker.rerank_without_model(
                    candidates=candidates,
                    analyzed_query=aq,
                    top_k=top_k,
                )
                reranker_used = False

        # ===== 构建响应 =====
        results = []
        for item in ranked[:top_k]:
            entity = item.get("entity", item)
            # 置信度: 优先用 reranker 给的 score，否则用 distance
            raw_score = item.get("_rerank_score", item.get("distance", 0.0))
            confidence = round(max(0.0, min(1.0, float(raw_score))), 4)
            results.append(RetrievalResult(
                chunk_id=entity.get("chunk_id", ""),
                text=entity.get("text", ""),
                score=item.get("distance", 0.0) if "distance" in item else 0.0,
                confidence=confidence,
                domain=entity.get("domain", ""),
                category=entity.get("category", ""),
                file_path=entity.get("file_path", ""),
                doc_number=entity.get("doc_number", ""),
                voltage_level=entity.get("voltage_level", ""),
                publish_level=entity.get("publish_level", ""),
                discipline=entity.get("discipline", ""),
                equipment_type=entity.get("equipment_type", ""),
                year=entity.get("year", 0),
                region=entity.get("region", ""),
                page_num=entity.get("page_num", 0),
                is_drawing=entity.get("is_drawing", False),
            ))

        elapsed = (time.time() - t_start) * 1000

        # 重排序模型改为常驻，不再用后即卸：
        #   Windows 上 PyTorch 反复卸载/重载 CUDA 模型会触发 access violation 段错误
        #   (第二次请求重载 reranker 时崩溃)。RTX 4070 SUPER 12.9GB 下
        #   embedder(2GB)+reranker(2GB)+OCR按需(1.5GB) ≈ 5.5GB，余量充足，无需让显存。

        return SearchResponse(
            query=query,
            query_type=aq.query_type,
            domain=aq.domain,
            results=results,
            total_candidates=len(candidates),
            elapsed_ms=elapsed,
            filter_applied=aq.filter_expr,
            expanded_terms=aq.expanded_terms,
            coarse_results=coarse_snapshot,
        )

    def search_cross_domain(self, query: str, top_k: int = None) -> Dict[str, SearchResponse]:
        """
        跨域对比检索 — 并行搜索多个域
        """
        aq = self.analyzer.analyze(query)

        available_domains = list(self.config["domain_keywords"].keys())
        domains = list(dict.fromkeys(aq.parallel_domains))
        # 对比提示词是双域结构：显式识别不足两个时，
        # 从已配置域中补齐，且始终去重。
        for domain in available_domains:
            if domain not in domains:
                domains.append(domain)
            if len(domains) >= 2:
                break

        results = {}
        for domain in domains[:2]:
            results[domain] = self.search(query, top_k=top_k, domain_filter=domain)

        return results

    def get_document_by_number(self, doc_number: str) -> List[RetrievalResult]:
        """按文档编号精确查找"""
        aq = AnalyzedQuery(
            original_query=doc_number,
            doc_number=doc_number,
            query_type="document_lookup",
            filter_expr=f'doc_number like "%{doc_number}%"'
        )

        dense_vec, sparse_vec = self.embedder.encode_query(doc_number)
        candidates = self.store.hybrid_search(
            dense_vector=dense_vec,
            sparse_vector=sparse_vec,
            filter_expr=aq.filter_expr,
            limit=100,  # 同一文档可能有很多 chunk
        )
        candidates = self._merge_bm25_candidates(
            query=doc_number,
            candidates=candidates,
            filter_expr=aq.filter_expr,
            limit=100,
        )

        results = []
        for item in candidates:
            entity = item.get("entity", item)
            results.append(RetrievalResult(
                chunk_id=entity.get("chunk_id", ""),
                text=entity.get("text", ""),
                score=item.get("distance", 0.0),
                domain=entity.get("domain", ""),
                category=entity.get("category", ""),
                file_path=entity.get("file_path", ""),
                doc_number=entity.get("doc_number", ""),
                voltage_level=entity.get("voltage_level", ""),
                publish_level=entity.get("publish_level", ""),
                page_num=entity.get("page_num", 0),
            ))

        return results

    def format_context_for_llm(self, results: List[RetrievalResult],
                               max_chunks: int = 15) -> str:
        """按 chunk 去重并限制单文件占比后格式化 LLM 上下文。"""
        deduped = select_context_chunks(
            results,
            max_chunks=max_chunks,
            max_chunks_per_file=int(
                self.retrieval_config.get("max_chunks_per_file", 3)
            ),
        )

        context_parts = []
        for result in deduped:
            context_parts.append(result.to_context_str())

        return "\n\n".join(context_parts)

    # ===== 文件注册表识别: 完整文档注入 =====

    def detect_file_in_query(self, query: str) -> Optional[FileMatchResult]:
        """
        检测查询中是否引用了已注册的特定文件

        Args:
            query: 用户查询

        Returns:
            最佳匹配结果, 无匹配时返回 None
        """
        matches = self.file_registry.detect_files_in_query(query, min_score=0.5)
        if not matches:
            return None
        # 返回最高分的匹配
        return matches[0]

    def get_full_document(self, file_path: str = None,
                          file_hash: str = None) -> str:
        """
        获取文件的完整内容（所有 chunks 按页码排序后拼接）

        Args:
            file_path: 文件路径（与 Milvus 中存储的 file_path 匹配）
            file_hash: 文件哈希（优先使用，通过 chunk_id 前缀匹配）

        Returns:
            格式化的完整文档文本，若未找到则返回空字符串
        """
        chunks = []

        # The edited layout is the authoritative source for file-scoped
        # context. This prevents a stale Milvus generation from being injected
        # into the prompt even when a caller names the document explicitly.
        if file_hash:
            layout_pages = self._current_layout_pages(file_hash)
            if layout_pages is not None:
                chunks = [
                    {"page_num": page_num, "text": text}
                    for page_num, text in sorted(layout_pages.items())
                ]

        # 策略1: 按 file_hash 精确查询（最可靠，chunk_id 前缀匹配）
        if file_hash and not chunks:
            chunks = self.store.query_by_file_hash(file_hash, sort_by_page=True)

        # 策略2: 按 file_path 查询
        if not chunks and file_path:
            chunks = self.store.query_by_file_path(file_path, sort_by_page=True)

        # 策略3: 从注册表获取 hash 后再查
        if not chunks and file_path:
            entry = self.file_registry.get_entry_by_filename(
                os.path.basename(file_path)
            )
            if entry and entry.file_hash:
                chunks = self.store.query_by_file_hash(
                    entry.file_hash, sort_by_page=True
                )

        if not chunks:
            return ""

        display_path = file_path or (file_hash or "unknown")
        return self._format_full_document(chunks, display_path)

    def _format_full_document(self, chunks: List[dict],
                              file_path: str) -> str:
        """
        格式化完整文档内容

        输出格式:
        【完整文档: {文件名}】（共 N 个片段, M 页）
        --- 第 1 页 ---
        ...chunk text...
        --- 第 2 页 ---
        ...chunk text...
        """
        import os
        fname = os.path.basename(file_path)

        # 统计页码
        pages = set()
        for c in chunks:
            p = c.get("page_num", 0) or 0
            if p > 0:
                pages.add(p)

        total_pages = max(pages) if pages else 0
        header = (
            f"\n\n{'=' * 60}\n"
            f"【完整文档: {fname}】"
            f"（共 {len(chunks)} 个片段"
        )
        if total_pages:
            header += f", {total_pages} 页"
        header += f"）\n{'=' * 60}\n"

        parts = [header]
        current_page = -1

        for chunk in chunks:
            page = chunk.get("page_num", 0) or 0
            if page > 0 and page != current_page:
                current_page = page
                parts.append(f"\n--- 第 {page} 页 ---\n")

            text = chunk.get("text", "")
            if text.strip():
                parts.append(text)

        parts.append(f"\n{'=' * 60}\n")

        return "\n".join(parts)

    def build_context_with_file_injection(
        self,
        query: str,
        search_results: List[RetrievalResult],
        max_chunks: int = 15,
    ) -> Tuple[str, Optional[FileMatchResult]]:
        """
        构建 LLM 上下文，若 query 中引用了特定文件则注入完整文档

        策略:
        1. 检测 query 中的文件名引用
        2. 若匹配到文件: 完整文档放在最前面，检索结果作为补充
        3. 若未匹配: 正常使用检索结果

        Args:
            query: 用户查询
            search_results: 正常检索结果
            max_chunks: 最大检索 chunk 数

        Returns:
            (context_text, matched_file_or_None)
        """
        file_match = self.detect_file_in_query(query)

        if not file_match:
            # 无文件名匹配，正常流程
            print(f"[file-registry] 查询中未检测到文件名: {query[:60]}")
            return self.format_context_for_llm(search_results, max_chunks), None

        # 有文件名匹配: 获取完整文档
        file_path = (file_match.entry.original_path or
                     file_match.entry.stored_path or
                     file_match.entry.file_name)

        print(f"[file-registry] 检测到文件名: {file_match.entry.file_name} "
              f"(score={file_match.match_score}, type={file_match.match_type})")

        # 优先用 file_hash 查询（chunk_id 前缀匹配，最可靠）
        full_doc = self.get_full_document(
            file_path=file_path,
            file_hash=file_match.entry.file_hash,
        )

        if not full_doc:
            # 文件在注册表中但向量库中无数据，回退正常流程
            print(f"[file-registry] WARN: 完整文档检索失败, "
                  f"file_hash={file_match.entry.file_hash}, "
                  f"回退到常规检索 (可能返回多个文件)")
            return self.format_context_for_llm(search_results, max_chunks), file_match

        print(f"[file-registry] 完整文档已注入: {len(full_doc)} 字符, "
              f"源文件: {file_match.entry.file_name}")

        # 上下文预算检查: 完整文档超长时截断，防止撑爆 LLM 窗口
        # Qwen3.5:4b num_ctx=32768, 中文字符≈0.5 token → 安全上限10000 chars (约5000 tokens)
        # 留出 prompt template (~500) + generation budget (4096) = 足够的空间
        MAX_FULL_DOC_CHARS = 10000
        if len(full_doc) > MAX_FULL_DOC_CHARS:
            orig_len = len(full_doc)
            full_doc = full_doc[:MAX_FULL_DOC_CHARS] + (
                f"\n\n【注意：完整文档过长（{orig_len}字符），已截断至{MAX_FULL_DOC_CHARS}字符。"
                f"此处仅展示前半部分，如需后续内容请具体提问。】"
            )
            print(f"[file-registry] WARN: 完整文档过长 ({orig_len} 字符), "
                  f"截断至 {MAX_FULL_DOC_CHARS} 字符")

        # 构建注入式上下文: 完整文档 + 聚焦指令 + 检索补充
        context_parts = []

        # 主文档（完整内容）
        context_parts.append(full_doc)

        # 聚焦指令
        fname = file_match.entry.file_name
        context_parts.append(
            f"\n【⚠ 重要指令: 用户查询引用了文件 \"{fname}\"，"
            f"请只基于上述完整文档内容回答，不要引用其他文件。"
            f"以下其他文件片段仅供背景了解，回答中不要引用它们的内容。】\n"
        )

        # 补充检索结果（去重，排除主文件 + 排除同系列文件）
        series_key = self._extract_series_key(fname)
        supplementary = []
        excluded_same_series = 0
        seen_files = {file_path}
        for r in search_results[:max_chunks]:
            if not r.file_path:
                continue
            if r.file_path in seen_files:
                continue
            # 排除与匹配文件同系列的其他文件
            if series_key and self._extract_series_key(
                os.path.basename(r.file_path)
            ) == series_key:
                excluded_same_series += 1
                continue
            seen_files.add(r.file_path)
            supplementary.append(r)

        print(f"[file-registry] 补充检索: {len(supplementary)} 文件, "
              f"同系列已排除: {excluded_same_series}")


        if supplementary:
            context_parts.append(
                f"\n{'─' * 40}\n"
                f"【以下为检索到的其他相关文件片段，仅供参考背景，"
                f"回答时请勿引用以下内容】\n"
                f"{'─' * 40}\n"
            )
            # 最多保留2个补充文件（大幅减少干扰）
            for r in supplementary[:2]:
                context_parts.append(r.to_context_str())

        return "\n\n".join(context_parts), file_match

    def search_statistical(self, query: str, aq: AnalyzedQuery) -> StatisticalResult:
        """统计型查询: 全量元数据聚合而非语义搜索 top-k

        流程:
          1. 从查询中提取统计维度
          2. 用 Milvus query() 标量过滤获取全量 chunk 元数据
          3. Python 聚合: Counter 分组计数, set 去重文件
          4. 格式化为 LLM 可用的结构化文本

        Args:
            query: 用户原始查询
            aq: 已分析的查询 (含 domain_filter)

        Returns:
            StatisticalResult with formatted_table
        """
        t0 = time.time()
        from collections import Counter
        from ingestion.milvus_store import build_filter_expression

        # 提取统计维度
        dims = self.analyzer._extract_statistical_dimensions(query)

        # 判断统计类型: count / list / distribution
        if any(kw in query for kw in ["哪些", "列出", "有哪些"]):
            stats_type = "list"
        elif any(kw in query for kw in ["分布", "占比", "分组", "按"]):
            stats_type = "distribution"
        else:
            stats_type = "count"

        # 构建过滤器 (仅 domain)
        filter_expr = build_filter_expression(domain=aq.domain)

        # 输出字段: 只取需要的标量字段
        output_fields = [
            "chunk_id", "file_path", "file_type",
            "domain", "category", "voltage_level",
            "publish_level", "discipline", "equipment_type",
            "year",
        ]

        # === 全量分页 + 在线聚合（不保留固定 20,000 行上限） ===
        domain_counter = Counter()
        cat_counter = Counter()
        volt_counter = Counter()
        pub_counter = Counter()
        disc_counter = Counter()
        year_counter = Counter()
        seen_files = set()
        file_chunk_counts = Counter()
        total_chunks = 0
        try:
            for batch in self.store.iter_scalar_query(
                filter_expr or "", output_fields, batch_size=1000
            ):
                for row in batch:
                    entity = row.get("entity", row)
                    total_chunks += 1
                    domain_counter[entity.get("domain", "") or "未分类"] += 1
                    cat_counter[entity.get("category", "") or "未分类"] += 1
                    volt_counter[entity.get("voltage_level", "") or "未指定"] += 1
                    pub_counter[entity.get("publish_level", "") or "未指定"] += 1
                    disc_counter[entity.get("discipline", "") or "未指定"] += 1
                    year = entity.get("year", 0)
                    if year:
                        year_counter[str(year)] += 1
                    fp = entity.get("file_path", "")
                    if fp:
                        seen_files.add(fp)
                        file_chunk_counts[fp] += 1
        except Exception as e:
            print(f"[statistical] Milvus paged query failed: {e}")
            return StatisticalResult(
                query_type=stats_type,
                total_chunks=0,
                unique_files=0,
                formatted_table=(
                    "[统计查询失败] 无法保证全量读取，未返回可能不完整的统计值。"
                ),
            )

        if total_chunks == 0:
            return StatisticalResult(
                query_type=stats_type,
                total_chunks=0,
                unique_files=0,
                formatted_table=f"## 统计结果\n\n未找到匹配条件" +
                    (f" (域: {aq.domain})" if aq.domain else "") +
                    " 的记录。\n",
            )

        unique_files = len(seen_files)

        # 构建文件列表 (去重, 按 chunk 数降序)
        file_entries = []
        for fp in sorted(seen_files, key=lambda f: file_chunk_counts.get(f, 0), reverse=True):
            file_entries.append({
                "name": fp.split("\\")[-1].split("/")[-1] if fp else "?",
                "chunks": file_chunk_counts.get(fp, 0),
            })

        result = StatisticalResult(
            query_type=stats_type,
            total_chunks=total_chunks,
            unique_files=unique_files,
            by_domain=dict(domain_counter.most_common()),
            by_category=dict(cat_counter.most_common()),
            by_voltage_level=dict(volt_counter.most_common()),
            by_publish_level=dict(pub_counter.most_common()),
            by_discipline=dict(disc_counter.most_common()),
            by_year=dict(sorted(year_counter.items())),
            file_list=file_entries,
        )
        result.formatted_table = self._format_stats_for_llm(
            result, query, aq.domain, dims, stats_type
        )

        print(f"[statistical] {stats_type} query: {total_chunks} chunks, "
              f"{unique_files} files, {(time.time() - t0) * 1000:.0f}ms")
        return result

    def _format_stats_for_llm(
        self,
        result: StatisticalResult,
        query: str,
        domain: str,
        dims: dict,
        stats_type: str,
    ) -> str:
        """将聚合结果格式化为 LLM 可用的 Markdown 统计表"""
        parts = [f"## 知识库统计结果\n"]

        domain_hint = f"（域过滤: {domain}）" if domain else "（全库）"
        parts.append(f"**查询**: {query} {domain_hint}")
        parts.append(f"**总计**: {result.total_chunks} 个文本块，"
                     f"来自 {result.unique_files} 个文件\n")
        parts.append("_统计口径：已通过分页读取全部匹配记录。_\n")

        # 按域分布 (如果有多域或明确要求)
        if dims.get("domain") or len(result.by_domain) > 1:
            parts.append("### 按专业域分布")
            parts.append("| 域 | 文本块数 |")
            parts.append("|---|---|")
            for k, v in result.by_domain.items():
                parts.append(f"| {k} | {v} |")
            parts.append("")

        # 按类目分布
        if dims.get("category") or len(result.by_category) > 1:
            parts.append("### 按类目分布")
            parts.append("| 类目 | 文本块数 |")
            parts.append("|---|---|")
            for k, v in result.by_category.items():
                parts.append(f"| {k} | {v} |")
            parts.append("")

        # 按电压等级
        if dims.get("voltage_level") or len(result.by_voltage_level) > 1:
            parts.append("### 按电压等级分布")
            parts.append("| 电压等级 | 文本块数 |")
            parts.append("|---|---|")
            for k, v in result.by_voltage_level.items():
                parts.append(f"| {k} | {v} |")
            parts.append("")

        # 按发布层级
        if dims.get("publish_level") or len(result.by_publish_level) > 1:
            parts.append("### 按发布层级分布")
            parts.append("| 发布层级 | 文本块数 |")
            parts.append("|---|---|")
            for k, v in result.by_publish_level.items():
                parts.append(f"| {k} | {v} |")
            parts.append("")

        # 按专业
        if dims.get("discipline") or len(result.by_discipline) > 1:
            parts.append("### 按专业类型分布")
            parts.append("| 专业类型 | 文本块数 |")
            parts.append("|---|---|")
            for k, v in result.by_discipline.items():
                parts.append(f"| {k} | {v} |")
            parts.append("")

        # 文件列表 (列举型或分布型显示前 30 个)
        if stats_type in ("list", "distribution") and result.file_list:
            show_n = min(30, len(result.file_list))
            parts.append(f"### 文件列表 (前 {show_n} 个，按内容量排序)")
            parts.append("| # | 文件名 | 文本块数 |")
            parts.append("|---|---|---|")
            for i, fe in enumerate(result.file_list[:show_n]):
                parts.append(f"| {i + 1} | {fe['name'][:80]} | {fe['chunks']} |")
            if len(result.file_list) > show_n:
                parts.append(f"| ... | 还有 {len(result.file_list) - show_n} 个文件 | ... |")
            parts.append("")

        # 计数型只显示文件数量
        if stats_type == "count":
            parts.append(f"### 文件清单")
            if result.file_list:
                parts.append(f"共 {len(result.file_list)} 个文件:")
                for i, fe in enumerate(result.file_list[:15]):
                    parts.append(f"- {fe['name'][:120]}")
                if len(result.file_list) > 15:
                    parts.append(f"- ... 还有 {len(result.file_list) - 15} 个文件")
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _extract_series_key(filename: str) -> Optional[str]:
        """
        从文件名中提取"系列标识"。
        同系列的文件共享相同的关键词+序号模板，如：
          "01会议材料之一2022年..." → "会议材料之"
          "02会议材料之二2023年..." → "会议材料之"
        用于在文件匹配时排除同系列的其他文件干扰。

        Returns:
            系列标识字符串, 无匹配时返回 None
        """
        return extract_series_key(filename)


# 快速测试
if __name__ == "__main__":
    # 测试检索管道（需先构建索引）
    retriever = Retriever()

    test_queries = [
        "变电消防设计要求",
        "10kV配电安全距离",
        "变压器接地保护",
    ]

    for q in test_queries:
        print(f"\n{'=' * 60}")
        print(f"🔍 {q}")
        print(f"{'=' * 60}")

        try:
            response = retriever.search(q, top_k=5)
            print(f"   类型: {response.query_type}")
            print(f"   域: {response.domain}")
            print(f"   过滤: {response.filter_applied}")
            print(f"   候选数: {response.total_candidates}")
            print(f"   耗时: {response.elapsed_ms:.0f}ms")
            print(f"\n   Top-3 结果:")
            for i, r in enumerate(response.results[:3]):
                print(f"   [{i + 1}] {r.doc_number} | {r.domain}/{r.category} | {r.file_path}")
                print(f"       预览: {r.text[:100]}...")
        except Exception as e:
            print(f"   ⚠ 错误: {e}")
