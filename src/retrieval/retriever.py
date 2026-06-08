"""
检索编排器 — 三阶段检索管道的顶层协调器
阶段0: 查询分析
阶段1: 粗召回 (混合搜索)
阶段2: 精排 (交叉编码器 + 元数据)
"""

import sys
import os
import time
import yaml
from typing import List, Dict, Optional
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieval.query_analyzer import QueryAnalyzer, AnalyzedQuery
from retrieval.reranker import Reranker
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
        """格式化为 LLM 上下文"""
        meta_parts = []
        if self.doc_number:
            meta_parts.append(f"文件编号: {self.doc_number}")
        if self.domain:
            meta_parts.append(f"专业域: {self.domain}")
        if self.category:
            meta_parts.append(f"文档类目: {self.category}")
        if self.voltage_level:
            meta_parts.append(f"电压等级: {self.voltage_level}")
        if self.publish_level:
            meta_parts.append(f"发布层级: {self.publish_level}")
        if self.page_num:
            meta_parts.append(f"页码: {self.page_num}")
        if self.file_path:
            meta_parts.append(f"来源: {self.file_path}")

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
        else:
            try:
                ranked = self.reranker.rerank(
                    query=query,
                    candidates=candidates,
                    analyzed_query=aq,
                    top_k=top_k,
                )
            except Exception as e:
                print(f"[warn] 重排序失败，回退元数据排序: {e}")
                ranked = self.reranker.rerank_without_model(
                    candidates=candidates,
                    analyzed_query=aq,
                    top_k=top_k,
                )

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
        """将检索结果格式化为 LLM 上下文"""
        chunks = results[:max_chunks]
        context_parts = []

        for i, result in enumerate(chunks):
            context_parts.append(
                f"--- 参考资料 {i + 1} ---\n"
                f"{result.to_context_str()}"
            )

        return "\n\n".join(context_parts)


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
