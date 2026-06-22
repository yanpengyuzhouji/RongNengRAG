"""
评估指标 — RAG 检索三层评估框架

所有函数为纯函数（无副作用），操作简单数据类型。
纯 Python 实现，不依赖 numpy。

Layer 1: 检索质量 (混合搜索阶段，粗排候选池)
Layer 2: 召回质量 (文档/Chunk层面)
Layer 3: 重排序效果 (粗排→精排对比)
"""

import math
import os
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class Layer1Metrics:
    """检索质量指标 (在粗排候选池上计算, K=50)"""
    recall_at_10: float = 0.0
    recall_at_30: float = 0.0
    recall_at_50: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    mrr: float = 0.0               # Mean Reciprocal Rank (chunk级别)
    ndcg_at_10: float = 0.0
    hit_rate: float = 0.0          # 至少命中1个相关chunk


@dataclass
class Layer2Metrics:
    """召回质量指标 (文档/Chunk层面)"""
    doc_recall: float = 0.0        # 候选池命中预期文档数 / 总预期文档数
    chunk_recall: float = 0.0      # 命中的相关chunk / 标注的全部相关chunk
    cross_doc_coverage_pct: float = 0.0  # 候选池中不同来源文档的覆盖比例
    domain_accuracy: float = 0.0   # Top结果domain匹配率
    category_accuracy: float = 0.0 # Top结果category匹配率


@dataclass
class Layer3Metrics:
    """重排序效果指标 (粗排 vs 精排对比)"""
    top1_improvement: float = 0.0  # 正确答案从非Top1提升到Top1的题目比例
    mrr_delta: float = 0.0         # MRR(精排) - MRR(粗排)
    ndcg_delta: float = 0.0        # NDCG(精排) - NDCG(粗排)
    degradation_count: int = 0     # 重排后退化的题目数
    coarse_top1_rank: Optional[int] = None   # 粗排中第一个相关chunk的排名 (-1表示未找到)
    fine_top1_rank: Optional[int] = None     # 精排中第一个相关chunk的排名


@dataclass
class PerQuestionMetrics:
    """单题全部指标"""
    question_id: int
    question: str

    # Ground truth
    expected_keywords: List[str] = field(default_factory=list)
    expected_top1_doc: str = ""
    relevant_chunks: List[str] = field(default_factory=list)

    # 粗排结果 (pre-reranker)
    coarse_chunk_ids: List[str] = field(default_factory=list)
    coarse_file_names: List[str] = field(default_factory=list)
    coarse_scores: List[float] = field(default_factory=list)

    # 精排结果 (post-reranker)
    fine_chunk_ids: List[str] = field(default_factory=list)
    fine_file_names: List[str] = field(default_factory=list)
    fine_scores: List[float] = field(default_factory=list)
    fine_confidences: List[float] = field(default_factory=list)

    # 三层指标
    layer1: Layer1Metrics = field(default_factory=Layer1Metrics)
    layer2: Layer2Metrics = field(default_factory=Layer2Metrics)
    layer3: Layer3Metrics = field(default_factory=Layer3Metrics)

    # 关键词命中 (兼容旧版)
    keyword_hits: Dict[str, bool] = field(default_factory=dict)
    keyword_hit_rate: float = 0.0

    # 元数据
    elapsed_ms: float = 0.0
    total_candidates: int = 0
    domain_filter: str = ""
    relevant_ids_used: Set[str] = field(default_factory=set)


# ============================================================
# 基础指标函数
# ============================================================

def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Recall@K = |relevant ∩ retrieved[:k]| / |relevant|"""
    if not relevant_ids:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    return len(retrieved_set & relevant_ids) / len(relevant_ids)


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Precision@K = |relevant ∩ retrieved[:k]| / k"""
    if k <= 0:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    return len(retrieved_set & relevant_ids) / k


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """MRR = 1 / rank_of_first_relevant, 无命中返回0"""
    if not relevant_ids:
        return 0.0
    for i, cid in enumerate(retrieved_ids):
        if cid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """NDCG@K, 二元相关性 (1 if in relevant_ids else 0)"""
    if not relevant_ids or k <= 0:
        return 0.0

    # DCG
    dcg = 0.0
    for i, cid in enumerate(retrieved_ids[:k]):
        if cid in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)  # i+2 因为 i 从0开始

    # IDCG: 所有相关chunk排在前面
    ideal_relevance = [1.0] * min(len(relevant_ids), k) + [0.0] * max(0, k - len(relevant_ids))
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """至少命中一个相关chunk返回1.0，否则0.0"""
    if not relevant_ids:
        return 0.0
    return 1.0 if any(cid in relevant_ids for cid in retrieved_ids) else 0.0


def find_first_relevant_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> Optional[int]:
    """找到第一个相关chunk的排名 (1-based)，无命中返回None"""
    if not relevant_ids:
        return None
    for i, cid in enumerate(retrieved_ids):
        if cid in relevant_ids:
            return i + 1
    return None


# ============================================================
# 相关性推导
# ============================================================

def compute_relevance_ids_from_keywords(
    candidates: List[Dict[str, str]],
    keywords: List[str],
) -> Set[str]:
    """
    从关键词在候选池文本中匹配来推导相关chunk_id。

    当 relevant_chunks 为空时使用此方法。
    一个chunk包含任一关键词即视为相关。
    """
    if not keywords:
        return set()

    relevant = set()
    for c in candidates:
        text = c.get("text", "")
        chunk_id = c.get("chunk_id", "")
        if not chunk_id:
            continue
        # 大小写不敏感的匹配
        text_lower = text.lower()
        for kw in keywords:
            if kw.lower() in text_lower:
                relevant.add(chunk_id)
                break
    return relevant


def compute_keyword_hits(text: str, keywords: List[str]) -> Dict[str, bool]:
    """检查每个关键词是否出现在合并文本中。"""
    text_lower = text.lower()
    return {kw: kw.lower() in text_lower for kw in keywords}


# ============================================================
# 三层指标计算
# ============================================================

def compute_layer1(
    coarse_chunk_ids: List[str],
    relevant_ids: Set[str],
) -> Layer1Metrics:
    """从粗排候选列表计算 Layer 1 检索质量指标。"""
    return Layer1Metrics(
        recall_at_10=recall_at_k(coarse_chunk_ids, relevant_ids, 10),
        recall_at_30=recall_at_k(coarse_chunk_ids, relevant_ids, 30),
        recall_at_50=recall_at_k(coarse_chunk_ids, relevant_ids, 50),
        precision_at_5=precision_at_k(coarse_chunk_ids, relevant_ids, 5),
        precision_at_10=precision_at_k(coarse_chunk_ids, relevant_ids, 10),
        mrr=mean_reciprocal_rank(coarse_chunk_ids, relevant_ids),
        ndcg_at_10=ndcg_at_k(coarse_chunk_ids, relevant_ids, 10),
        hit_rate=hit_rate(coarse_chunk_ids, relevant_ids),
    )


def compute_layer2(
    coarse_results: List[Dict],
    coarse_chunk_ids: List[str],
    relevant_ids: Set[str],
    expected_top1_doc: str,
    expected_domain: str,
    expected_category: str,
) -> Layer2Metrics:
    """计算 Layer 2 召回质量指标。"""
    # 文档召回率: 候选池中出现了多少不同的源文档
    if expected_top1_doc:
        unique_docs_in_pool = set()
        for c in coarse_results:
            fp = c.get("file_path", "")
            if fp:
                unique_docs_in_pool.add(os.path.basename(fp))
        expected_docs = {expected_top1_doc} if expected_top1_doc else set()
        doc_recall = len(unique_docs_in_pool & expected_docs) / len(expected_docs) if expected_docs else 0.0
    else:
        doc_recall = 0.0

    # Chunk 召回率: 相关chunk在候选池中的占比
    if relevant_ids:
        relevant_in_pool = set(coarse_chunk_ids) & relevant_ids
        chunk_recall = len(relevant_in_pool) / len(relevant_ids)
    else:
        chunk_recall = 0.0

    # 跨文档覆盖率: 候选池中不重复文档数
    all_docs_in_pool = set()
    for c in coarse_results:
        fp = c.get("file_path", "")
        if fp:
            all_docs_in_pool.add(os.path.basename(fp))
    # 保守估计，至少记录了1个文档
    cross_doc_coverage_pct = min(len(all_docs_in_pool) / max(len(coarse_results), 1) * 100, 100.0)

    # 域/类目准确率: Top结果的domain和category是否匹配
    domain_match = False
    category_match = False
    if coarse_results:
        top = coarse_results[0]
        if expected_domain:
            domain_match = top.get("domain", "") == expected_domain
        if expected_category:
            category_match = top.get("category", "") == expected_category

    return Layer2Metrics(
        doc_recall=doc_recall,
        chunk_recall=chunk_recall,
        cross_doc_coverage_pct=cross_doc_coverage_pct,
        domain_accuracy=1.0 if domain_match else 0.0,
        category_accuracy=1.0 if category_match else 0.0,
    )


def compute_layer3(
    coarse_chunk_ids: List[str],
    fine_chunk_ids: List[str],
    relevant_ids: Set[str],
) -> Layer3Metrics:
    """计算 Layer 3 重排序效果指标。"""
    if not relevant_ids:
        return Layer3Metrics()

    # 粗排/精排中第一个相关chunk的排名
    coarse_rank = find_first_relevant_rank(coarse_chunk_ids, relevant_ids)
    fine_rank = find_first_relevant_rank(fine_chunk_ids, relevant_ids)

    # Top-1 改进: 粗排时正确答案不在Top1，精排后到了Top1
    top1_impr = 0.0
    if coarse_rank is not None and fine_rank is not None:
        if coarse_rank > 1 and fine_rank == 1:
            top1_impr = 1.0

    # MRR
    mrr_before = mean_reciprocal_rank(coarse_chunk_ids, relevant_ids)
    mrr_after = mean_reciprocal_rank(fine_chunk_ids, relevant_ids)
    mrr_delta = mrr_after - mrr_before

    # NDCG@10
    ndcg_before = ndcg_at_k(coarse_chunk_ids, relevant_ids, 10)
    ndcg_after = ndcg_at_k(fine_chunk_ids, relevant_ids, 10)
    ndcg_delta = ndcg_after - ndcg_before

    # 退化检测: 粗排有命中 + 精排排名比粗排更差
    degradation_count = 0
    if coarse_rank is not None and fine_rank is not None and fine_rank > coarse_rank:
        degradation_count = 1

    return Layer3Metrics(
        top1_improvement=top1_impr,
        mrr_delta=mrr_delta,
        ndcg_delta=ndcg_delta,
        degradation_count=degradation_count,
        coarse_top1_rank=coarse_rank,
        fine_top1_rank=fine_rank,
    )


# ============================================================
# 主编排函数
# ============================================================

def compute_all_metrics(
    question_data: Dict,
    coarse_results: List[Dict],
    fine_results: List[Dict],
    elapsed_ms: float,
    relevant_ids_override: Optional[Set[str]] = None,
) -> PerQuestionMetrics:
    """
    计算单题的所有三层指标。

    Args:
        question_data: 数据集中的题目 dict (含 id, question, expected_keywords 等)
        coarse_results: 粗排候选 list of dict (chunk_id, text, file_path, domain, category, score)
        fine_results: 精排结果 list of dict (同上, 另有 confidence)
        elapsed_ms: 检索耗时
        relevant_ids_override: 手动指定的相关chunk_id集合，为None时自动从关键词推导

    Returns:
        PerQuestionMetrics with all three layers populated
    """
    qid = question_data.get("id", 0)
    question = question_data.get("question", "")
    keywords = question_data.get("expected_keywords", [])
    expected_top1_doc = question_data.get("expected_top1_doc", "")
    domain_filter = question_data.get("domain_filter", "")
    category = question_data.get("category", "")

    # 提取ID列表
    coarse_ids = [c.get("chunk_id", "") for c in coarse_results]
    fine_ids = [c.get("chunk_id", "") for c in fine_results]

    # 相关性推导
    if relevant_ids_override is not None:
        relevant_ids = relevant_ids_override
    elif question_data.get("relevant_chunks"):
        relevant_ids = set(question_data["relevant_chunks"])
    else:
        relevant_ids = compute_relevance_ids_from_keywords(coarse_results, keywords)

    # Layer 1
    layer1 = compute_layer1(coarse_ids, relevant_ids)

    # Layer 2
    layer2 = compute_layer2(
        coarse_results=coarse_results,
        coarse_chunk_ids=coarse_ids,
        relevant_ids=relevant_ids,
        expected_top1_doc=expected_top1_doc,
        expected_domain=domain_filter,
        expected_category=category,
    )

    # Layer 3
    layer3 = compute_layer3(coarse_ids, fine_ids, relevant_ids)

    # 关键词命中 (兼容旧版)
    all_text = " ".join(c.get("text", "") for c in fine_results[:10])
    keyword_hits = compute_keyword_hits(all_text, keywords)
    kw_hit_count = sum(1 for v in keyword_hits.values() if v)
    kw_hit_rate = kw_hit_count / len(keywords) if keywords else 0.0

    return PerQuestionMetrics(
        question_id=qid,
        question=question,
        expected_keywords=keywords,
        expected_top1_doc=expected_top1_doc,
        relevant_chunks=list(relevant_ids),

        coarse_chunk_ids=coarse_ids,
        coarse_file_names=[c.get("file_path", "") for c in coarse_results],
        coarse_scores=[c.get("score", 0.0) for c in coarse_results],

        fine_chunk_ids=fine_ids,
        fine_file_names=[c.get("file_path", "") for c in fine_results],
        fine_scores=[c.get("score", 0.0) for c in fine_results],
        fine_confidences=[c.get("confidence", 0.0) for c in fine_results],

        layer1=layer1,
        layer2=layer2,
        layer3=layer3,

        keyword_hits=keyword_hits,
        keyword_hit_rate=kw_hit_rate,

        elapsed_ms=elapsed_ms,
        total_candidates=len(coarse_results),
        domain_filter=domain_filter,
        relevant_ids_used=relevant_ids,
    )


# ============================================================
# 聚合函数
# ============================================================

def aggregate_metrics(all_metrics: List[PerQuestionMetrics]) -> Dict[str, Any]:
    """跨所有题目计算宏平均指标。"""
    if not all_metrics:
        return {}

    n = len(all_metrics)

    def mean(values):
        return sum(values) / n if n > 0 else 0.0

    # Layer 1 聚合
    l1 = {
        "avg_recall_at_10": mean(m.layer1.recall_at_10 for m in all_metrics),
        "avg_recall_at_30": mean(m.layer1.recall_at_30 for m in all_metrics),
        "avg_recall_at_50": mean(m.layer1.recall_at_50 for m in all_metrics),
        "avg_precision_at_5": mean(m.layer1.precision_at_5 for m in all_metrics),
        "avg_precision_at_10": mean(m.layer1.precision_at_10 for m in all_metrics),
        "avg_mrr": mean(m.layer1.mrr for m in all_metrics),
        "avg_ndcg_at_10": mean(m.layer1.ndcg_at_10 for m in all_metrics),
        "avg_hit_rate": mean(m.layer1.hit_rate for m in all_metrics),
    }

    # Layer 2 聚合
    l2 = {
        "avg_doc_recall": mean(m.layer2.doc_recall for m in all_metrics),
        "avg_chunk_recall": mean(m.layer2.chunk_recall for m in all_metrics),
        "avg_cross_doc_coverage_pct": mean(m.layer2.cross_doc_coverage_pct for m in all_metrics),
        "domain_accuracy": mean(m.layer2.domain_accuracy for m in all_metrics),
        "category_accuracy": mean(m.layer2.category_accuracy for m in all_metrics),
    }

    # Layer 3 聚合
    l3 = {
        "top1_improvement_rate": mean(m.layer3.top1_improvement for m in all_metrics),
        "avg_mrr_delta": mean(m.layer3.mrr_delta for m in all_metrics),
        "avg_ndcg_delta": mean(m.layer3.ndcg_delta for m in all_metrics),
        "total_degradation_cases": sum(m.layer3.degradation_count for m in all_metrics),
    }

    # 关键词
    kw = {
        "avg_keyword_hit_rate": mean(m.keyword_hit_rate for m in all_metrics),
    }

    return {
        "num_questions": n,
        "layer1": l1,
        "layer2": l2,
        "layer3": l3,
        "keyword": kw,
    }


def aggregate_by_category(all_metrics: List[PerQuestionMetrics]) -> Dict[str, Dict]:
    """按 category 分组聚合。"""
    from collections import defaultdict
    groups = defaultdict(list)
    for m in all_metrics:
        # 从 question_data 没办法直接拿到 category，但我们可以从 dataset 方拿到
        # 这里从 layer2 的数据不够；需要在 run_eval 里获取
        pass
    # 这个函数会在 report_generator 里用到原数据
    return {}


# ============================================================
# 答案质量指标 (Answer Quality)
# ============================================================

@dataclass
class AnswerQualityMetrics:
    """端到端答案质量指标 (纯规则评分, 无LLM依赖)"""
    # 事实覆盖度
    fact_coverage: float = 0.0       # key_facts 匹配度均值 (0~1)
    fact_hit_count: int = 0          # 完全命中的事实点数 (score=1.0)
    fact_partial_count: int = 0      # 部分命中 (score=0.5)
    fact_miss_count: int = 0         # 未命中 (score=0.0)

    # 引用质量
    citation_precision: float = 0.0  # generated citations 命中 expected 的比例
    citation_recall: float = 0.0     # expected citations 被 generated 命中的比例

    # 文本质量
    rouge_l_f1: float = 0.0          # ROUGE-L F1
    rouge_l_precision: float = 0.0
    rouge_l_recall: float = 0.0
    answer_length: int = 0
    gold_length: int = 0
    length_ratio: float = 0.0        # answer/gold (理想 0.7~1.5)

    # 幻觉检测
    hallucination_count: int = 0     # 可能编造的规范编号数
    hallucination_details: List[str] = field(default_factory=list)


def _extract_searchable_parts(fact: str) -> List[str]:
    """从 fact 中提取可用于搜索的关键片段."""
    import re
    parts = []

    # 1. 按标点分割
    segments = re.split(r'[,，;；/、()（）\s]+', fact)
    for seg in segments:
        seg = seg.strip()
        if len(seg) >= 3:  # 至少3字符
            parts.append(seg)

    # 2. 提取规范编号的简短形式 (如 "GB 50545-2010" -> 也加入 "GB 50545")
    for seg in parts:
        m = re.match(r'([A-Z]{2,})\s*[-_]?\s*(\d+)', seg)
        if m:
            short = f"{m.group(1)} {m.group(2)}"
            if short not in parts:
                parts.append(short)

    # 3. 提取纯数字 (<4位) 和关键数值
    numbers = re.findall(r'[\d.]+[%kKkVΩmAW]?', fact)
    for n in numbers:
        if len(n) >= 3 and n not in parts:
            parts.append(n)

    if len(parts) < 2:
        return [fact.strip()]
    return parts


def compute_fact_coverage(answer: str, key_facts: List[str]) -> Tuple[float, int, int, int]:
    """
    计算答案对关键事实的覆盖度 (简化子串匹配).

    对每个 key_fact:
      - 拆分为关键片段
      - 在 answer 中检查每个片段是否出现
      - 全部出现 -> 1.0, 过半出现 -> 0.5, 否则 -> 0.0
    """
    answer_lower = answer.lower()

    hit = 0
    partial = 0
    miss = 0

    for fact in key_facts:
        parts = _extract_searchable_parts(fact)
        if len(parts) <= 1:
            # 整句匹配
            if fact.lower().strip() in answer_lower:
                hit += 1
            else:
                miss += 1
            continue

        matched = 0
        for p in parts:
            if p.lower() in answer_lower:
                matched += 1

        match_ratio = matched / len(parts)

        if match_ratio >= 0.6:
            hit += 1
        elif match_ratio >= 0.3:
            partial += 1
        else:
            miss += 1

    total = len(key_facts)
    coverage = (hit * 1.0 + partial * 0.5) / total if total > 0 else 0.0
    return coverage, hit, partial, miss


def compute_citation_accuracy(
    generated_citations: List[str],
    expected_citations: List[str],
) -> Tuple[float, float]:
    """
    计算引用准确率.

    对每个 expected citation, 检查是否有 generated citation 匹配 (子串包含).
    对每个 generated citation, 检查是否有 expected citation 匹配.

    Returns:
        (precision, recall)
    """
    if not expected_citations:
        return (0.0, 0.0)

    gen_lower = [c.lower().strip() for c in generated_citations]
    exp_lower = [c.lower().strip() for c in expected_citations]

    # Precision: generated 中有多少命中了 expected
    if generated_citations:
        hit_gen = sum(1 for g in gen_lower
                      if any(e in g or g in e for e in exp_lower))
        precision = hit_gen / len(generated_citations)
    else:
        # 如果没有生成引用，检查答案文本中是否直接出现了预期引用
        precision = 0.0

    # Recall: expected 中有多少被 generated 命中了
    hit_exp = sum(1 for e in exp_lower
                  if any(e in g or g in e for g in gen_lower))
    recall = hit_exp / len(expected_citations)

    return precision, recall


def compute_citation_accuracy_from_text(
    answer_text: str,
    expected_citations: List[str],
) -> Tuple[float, float, List[str]]:
    """
    从答案文本中直接检测引用 (不依赖 API 的 citations 提取).
    同时检查 【...】 格式和纯文本中的规范编号出现.
    """
    import re
    answer_lower = answer_text.lower()

    # 提取 【...】 中的内容
    bracket_refs = re.findall(r'[【\[].*?[】\]]', answer_text)
    all_refs = bracket_refs + [answer_text]  # 整体文本也检查

    if not expected_citations:
        return (0.0, 0.0, [])

    found = []
    for e in expected_citations:
        e_lower = e.lower().strip()
        matched = False
        for ref in all_refs:
            if e_lower in ref.lower():
                matched = True
                break
        # 也检查答案正文 (无 【】 包裹)
        if not matched and e_lower in answer_lower:
            matched = True
        found.append(e)

    exp_lower = [e.lower().strip() for e in expected_citations]
    hit_exp = sum(1 for e in exp_lower
                  if any(e in ref.lower() for ref in all_refs) or e in answer_lower)

    # Precision 简化为: 答案中实际包含了几个 expected citation
    precision = hit_exp / len(expected_citations) if expected_citations else 0.0
    recall = hit_exp / len(expected_citations) if expected_citations else 0.0

    return precision, recall, found


def detect_hallucination(
    answer: str,
    sources: List[dict],
) -> Tuple[int, List[str]]:
    """
    检测答案中是否引用了不存在的规范编号 (不在 sources 中).

    只检测明确的规范编号模式 (GB/T XXXX, JGJ XXXX 等),
    避免对普通数字和缩写误报.
    """
    import re
    # 只匹配明确的规范编号模式
    doc_patterns = re.findall(
        r'(?:GB|GB/T|GB\s*T|DL|DL/T|DL\s*T|JGJ|CJJ|CECS|Q/GDW|SL|NB|YS|SH)\s*\d+[-\d]*',
        answer, re.IGNORECASE
    )
    # 也匹配 【...】 中的内容
    bracket_refs = re.findall(r'[【\[].*?[】\]]', answer)

    # 从 sources 中提取已知的 doc_number
    known_docs = set()
    for s in sources:
        dn = s.get("doc_number", "")
        if dn:
            known_docs.add(dn.lower().strip().replace(" ", "").replace("_", ""))
        fp = s.get("file_path", "")
        if fp:
            # 从文件名提取可能的规范编号
            fn = fp.lower().replace("_", "").replace(" ", "")
            known_docs.add(fn)

    suspicious = []
    seen = set()
    for pat in doc_patterns:
        pat_norm = pat.lower().strip().replace(" ", "").replace("_", "")
        if pat_norm in seen:
            continue
        seen.add(pat_norm)
        # 检查是否在已知来源中 (模糊匹配)
        matched = False
        for kd in known_docs:
            if pat_norm in kd or kd in pat_norm or pat_norm[:6] in kd:
                matched = True
                break
        if not matched:
            suspicious.append(pat)

    return len(suspicious), suspicious


def _lcs_length(x: str, y: str) -> int:
    """最长公共子序列长度 (Longest Common Subsequence)."""
    if not x or not y:
        return 0
    # 使用两个一维数组节省内存
    prev = [0] * (len(y) + 1)
    curr = [0] * (len(y) + 1)
    for xi in x:
        for j, yj in enumerate(y):
            if xi == yj:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(curr[j], prev[j + 1])
        prev, curr = curr, prev
    return prev[len(y)]


def compute_rouge_l(reference: str, candidate: str) -> Tuple[float, float, float]:
    """
    计算 ROUGE-L (基于最长公共子序列).

    Returns:
        (f1, precision, recall)
    """
    ref_chars = reference.replace(" ", "")
    cand_chars = candidate.replace(" ", "")

    lcs_len = _lcs_length(ref_chars, cand_chars)

    precision = lcs_len / len(cand_chars) if cand_chars else 0.0
    recall = lcs_len / len(ref_chars) if ref_chars else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return f1, precision, recall


def compute_answer_quality(
    answer: str,
    gold_answer: str,
    key_facts: List[str],
    expected_citations: List[str],
    sources: List[dict],
) -> AnswerQualityMetrics:
    """
    计算端到端答案质量的所有指标 (纯规则, 无LLM依赖).
    """
    # 事实覆盖度
    fact_coverage, hit, partial, miss = compute_fact_coverage(answer, key_facts)

    # 引用准确率 (从答案文本直接提取)
    cit_prec, cit_rec, _ = compute_citation_accuracy_from_text(answer, expected_citations)

    # ROUGE-L
    rl_f1, rl_p, rl_r = compute_rouge_l(gold_answer, answer)

    # 答案长度
    ans_len = len(answer)
    gold_len = len(gold_answer)
    length_ratio = ans_len / gold_len if gold_len > 0 else 0.0

    # 幻觉检测
    hallu_count, hallu_details = detect_hallucination(answer, sources)

    return AnswerQualityMetrics(
        fact_coverage=fact_coverage,
        fact_hit_count=hit,
        fact_partial_count=partial,
        fact_miss_count=miss,
        citation_precision=cit_prec,
        citation_recall=cit_rec,
        rouge_l_f1=rl_f1,
        rouge_l_precision=rl_p,
        rouge_l_recall=rl_r,
        answer_length=ans_len,
        gold_length=gold_len,
        length_ratio=length_ratio,
        hallucination_count=hallu_count,
        hallucination_details=hallu_details,
    )
