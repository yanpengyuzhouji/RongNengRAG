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
import yaml
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

        self.retrieval_config = self.config["retrieval"]

    def search(self, query: str, top_k: int = None,
               domain_filter: str = None) -> SearchResponse:
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
        if domain_filter and not aq.domain:
            aq.domain = domain_filter
            aq.filter_expr = self.analyzer._build_filter_expr(aq)

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

        # 用后即卸：释放重排序模型 (~2GB 显存)，下次搜索时按需重载
        if reranker_used:
            self.reranker.unload()

        return SearchResponse(
            query=query,
            query_type=aq.query_type,
            domain=aq.domain,
            results=results,
            total_candidates=len(candidates),
            elapsed_ms=elapsed,
            filter_applied=aq.filter_expr,
            expanded_terms=aq.expanded_terms,
        )

    def search_cross_domain(self, query: str, top_k: int = None) -> Dict[str, SearchResponse]:
        """
        跨域对比检索 — 并行搜索多个域
        """
        aq = self.analyzer.analyze(query)

        if not aq.parallel_domains:
            # 自动选择可能相关的域
            aq.parallel_domains = list(self.config["domain_keywords"].keys())

        results = {}
        for domain in aq.parallel_domains[:3]:  # 最多3个域
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
        """将检索结果格式化为 LLM 上下文，按文件去重，不添加序号"""
        chunks = results[:max_chunks]
        seen = set()
        deduped = []
        for r in chunks:
            k = r.file_path or r.chunk_id
            if k not in seen:
                seen.add(k)
                deduped.append(r)

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

        # 策略1: 按 file_hash 精确查询（最可靠，chunk_id 前缀匹配）
        if file_hash:
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
        import re as _re
        # 会议材料系列: XX会议材料之X...
        m = _re.search(r'会议材料之', filename)
        if m:
            return '会议材料之'
        # 带有数字前缀+关键词的系列: "第X章" "第X部分" 等
        m = _re.search(r'(?:第([一二三四五六七八九十\d]+)[章节部分篇])', filename)
        if m:
            return m.group(2)  # "第X章" 等
        # 默认: 无系列标识，不排除
        return None


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
