"""
报告生成器 — Markdown + JSON 双格式输出
生成三层评估报告，风格对齐现有测试报告
"""

import json
import os
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from collections import defaultdict

from eval.metrics import PerQuestionMetrics, AnswerQualityMetrics, aggregate_metrics


def generate_report(
    all_metrics: List[PerQuestionMetrics],
    dataset: Dict,
    output_dir: str = "eval/output",
    answer_quality_list: Optional[List[AnswerQualityMetrics]] = None,
) -> Tuple[str, str]:
    """
    生成 Markdown 和 JSON 报告。

    Args:
        all_metrics: 所有题目的检索评估指标
        dataset: 数据集 dict
        output_dir: 输出目录
        answer_quality_list: 所有题目的答案质量指标 (可选)

    Returns:
        (md_path, json_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    q_meta = {}
    for q in dataset.get("questions", []):
        q_meta[q["id"]] = q

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(output_dir, f"eval_report_{timestamp}.md")
    json_path = os.path.join(output_dir, f"eval_results_{timestamp}.json")

    md_content = _build_markdown(all_metrics, q_meta, dataset, timestamp, answer_quality_list)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    json_content = _build_json(all_metrics, q_meta, dataset, timestamp, answer_quality_list)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_content, f, ensure_ascii=False, indent=2)

    return md_path, json_path


# ============================================================
# Markdown 生成
# ============================================================

def _build_markdown(
    all_metrics: List[PerQuestionMetrics],
    q_meta: Dict,
    dataset: Dict,
    timestamp: str,
    answer_quality_list: Optional[List[AnswerQualityMetrics]] = None,
) -> str:
    agg = aggregate_metrics(all_metrics)
    aq_list = answer_quality_list or []
    has_aq = bool(aq_list)
    lines = []

    # ---- 标题 ----
    lines.append(f"# RAG 检索评估报告 — 三层评估框架")
    lines.append("")
    lines.append(f"> 数据集: {dataset.get('dataset_name', '')}")
    lines.append(f"> 评估日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 题目数: {len(all_metrics)} | 粗召回候选: 50 | 精排 Top-K: 15")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- 一、评估概览 ----
    lines.append("## 一、评估概览")
    lines.append("")
    lines.append("| 层级 | 核心指标 | 数值 | 说明 |")
    lines.append("|------|----------|------|------|")
    l1 = agg.get("layer1", {})
    l2 = agg.get("layer2", {})
    l3 = agg.get("layer3", {})
    kw = agg.get("keyword", {})

    lines.append(f"| L1 检索质量 | Recall@50 | {l1.get('avg_recall_at_50', 0):.3f} | 粗排候选池50条中相关chunk覆盖率 |")
    lines.append(f"| L1 检索质量 | MRR | {l1.get('avg_mrr', 0):.3f} | 第一个相关chunk平均排名倒数 |")
    lines.append(f"| L1 检索质量 | NDCG@10 | {l1.get('avg_ndcg_at_10', 0):.3f} | 排序质量 (前10) |")
    lines.append(f"| L2 召回质量 | 文档召回率 | {l2.get('avg_doc_recall', 0):.3f} | 预期文档是否出现在候选池 |")
    lines.append(f"| L2 召回质量 | Domain准确率 | {l2.get('domain_accuracy', 0):.3f} | Top-1 domain是否匹配预期 |")
    lines.append(f"| L2 召回质量 | Category准确率 | {l2.get('category_accuracy', 0):.3f} | Top-1 category是否匹配预期 |")
    lines.append(f"| L3 重排序效果 | MRR Delta | {l3.get('avg_mrr_delta', 0):+.3f} | 精排后MRR变化 (正=改善) |")
    lines.append(f"| L3 重排序效果 | NDCG Delta | {l3.get('avg_ndcg_delta', 0):+.3f} | 精排后NDCG变化 |")
    lines.append(f"| L3 重排序效果 | 退化题数 | {l3.get('total_degradation_cases', 0)} | 重排后退化的题目数 |")
    lines.append(f"| 关键词 | 命中率 | {kw.get('avg_keyword_hit_rate', 0):.1%} | 预期关键词在Top-10结果中出现比例 |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- 二、Layer 1: 检索质量 ----
    lines.append("## 二、Layer 1: 检索质量 (混合搜索阶段)")
    lines.append("")
    lines.append("评估粗排候选池 (Top-50) 的覆盖能力和排序质量。")
    lines.append("")
    lines.append("| 指标 | 均值 | 中位数 | 最低 | 最高 |")
    lines.append("|------|------|--------|------|------|")

    for name, attr, k_val in [
        ("Recall@10", "recall_at_10", None),
        ("Recall@30", "recall_at_30", None),
        ("Recall@50", "recall_at_50", None),
        ("Precision@5", "precision_at_5", None),
        ("Precision@10", "precision_at_10", None),
        ("MRR", "mrr", None),
        ("NDCG@10", "ndcg_at_10", None),
        ("Hit Rate", "hit_rate", None),
    ]:
        values = [getattr(m.layer1, attr) for m in all_metrics]
        avg_val = sum(values) / len(values)
        med_val = sorted(values)[len(values) // 2]
        lines.append(f"| {name} | {avg_val:.4f} | {med_val:.4f} | {min(values):.4f} | {max(values):.4f} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- 三、Layer 2: 召回质量 ----
    lines.append("## 三、Layer 2: 召回质量 (文档/Chunk层面)")
    lines.append("")
    lines.append("评估系统从知识库中找回所有相关材料的能力。")
    lines.append("")
    lines.append("| 指标 | 均值 | 说明 |")
    lines.append("|------|------|------|")
    lines.append(f"| 文档召回率 | {l2.get('avg_doc_recall', 0):.3f} | 预期文档出现在候选池的比例 |")
    lines.append(f"| Chunk召回率 | {l2.get('avg_chunk_recall', 0):.3f} | 相关chunk被候选池覆盖的比例 |")
    lines.append(f"| 跨文档覆盖率 | {l2.get('avg_cross_doc_coverage_pct', 0):.1f}% | 候选池文档多样性 |")
    lines.append(f"| Domain准确率 | {l2.get('domain_accuracy', 0):.3f} | Top-1结果domain正确率 |")
    lines.append(f"| Category准确率 | {l2.get('category_accuracy', 0):.3f} | Top-1结果category正确率 |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- 四、Layer 3: 重排序效果 ----
    lines.append("## 四、Layer 3: 重排序效果 (Reranker)")
    lines.append("")
    lines.append("评估 BGE-Reranker-v2-m3 对最终排序质量的提升。")
    lines.append("")
    lines.append("| 指标 | 数值 | 说明 |")
    lines.append("|------|------|------|")
    lines.append(f"| Top-1 提升率 | {l3.get('top1_improvement_rate', 0):.1%} | 正确答案从非Top1提升到Top1 |")
    lines.append(f"| MRR Delta | {l3.get('avg_mrr_delta', 0):+.4f} | 正值表示改善 |")
    lines.append(f"| NDCG Delta | {l3.get('avg_ndcg_delta', 0):+.4f} | 正值表示排序更优 |")
    lines.append(f"| 退化题数 | {l3.get('total_degradation_cases', 0)}/{len(all_metrics)} | 重排后排名反而下降 |")

    # 找出退化案例
    degradation_cases = [m for m in all_metrics if m.layer3.degradation_count > 0]
    if degradation_cases:
        lines.append("")
        lines.append("### 退化案例")
        lines.append("")
        lines.append("| # | 问题 | 粗排排名 | 精排排名 | 退化幅度 |")
        lines.append("|---|------|----------|----------|----------|")
        for m in degradation_cases:
            cr = m.layer3.coarse_top1_rank or 0
            fr = m.layer3.fine_top1_rank or 0
            delta = fr - cr
            lines.append(f"| Q{m.question_id} | {m.question[:50]}... | {cr} | {fr} | +{delta} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- 五、答案质量 ----
    if has_aq:
        lines.append("## 五、答案质量 (端到端)")
        lines.append("")
        lines.append("评估 /ask API 生成的最终答案与金标答案的对比。")
        lines.append("")

        n_aq = len(aq_list)
        avg_fact = sum(a.fact_coverage for a in aq_list) / n_aq if n_aq else 0
        avg_cit_p = sum(a.citation_precision for a in aq_list) / n_aq if n_aq else 0
        avg_cit_r = sum(a.citation_recall for a in aq_list) / n_aq if n_aq else 0
        avg_rouge = sum(a.rouge_l_f1 for a in aq_list) / n_aq if n_aq else 0
        avg_len_ratio = sum(a.length_ratio for a in aq_list) / n_aq if n_aq else 0
        total_hallu = sum(a.hallucination_count for a in aq_list) if n_aq else 0

        lines.append("| 指标 | 均值 | 说明 |")
        lines.append("|------|------|------|")
        lines.append(f"| 事实覆盖度 | {avg_fact:.3f} | 金标key_facts被答案覆盖的比例 (0~1) |")
        lines.append(f"| 引用精确率 | {avg_cit_p:.3f} | 期望引用中在答案里出现过的比例 |")
        lines.append(f"| 引用召回率 | {avg_cit_r:.3f} | 期望引用被答案覆盖的比例 |")
        lines.append(f"| ROUGE-L F1 | {avg_rouge:.3f} | 答案与金标的长子序列相似度 |")
        lines.append(f"| 长度比 | {avg_len_ratio:.2f} | 生成答案/金标长度比 (理想0.7~1.5) |")
        lines.append(f"| 可疑引用 | {total_hallu} | 答案中可能编造的规范编号数 |")
        lines.append("")

        # 分类别答案质量
        lines.append("### 5.1 分类别答案质量")
        lines.append("")
        lines.append("| 类别 | 题目数 | Fact Cov | Cit Recall | ROUGE-L | 评级 |")
        lines.append("|------|--------|----------|------------|---------|------|")
        cat_aq = defaultdict(list)
        for m, aq in zip(all_metrics, aq_list):
            cat = q_meta.get(m.question_id, {}).get("category", "未知")
            cat_aq[cat].append(aq)
        for cat in sorted(cat_aq.keys()):
            items = cat_aq[cat]
            n = len(items)
            avg_f = sum(a.fact_coverage for a in items) / n
            avg_c = sum(a.citation_recall for a in items) / n
            avg_r = sum(a.rouge_l_f1 for a in items) / n
            stars = _stars(avg_f * 100)
            lines.append(f"| {cat} | {n} | {avg_f:.3f} | {avg_c:.3f} | {avg_r:.3f} | {stars} |")
        lines.append("")
        lines.append("---")
        lines.append("")

    # ---- 六、分类别明细（检索质量） ----
    lines.append("## 六、分类别明细 (检索质量)")
    lines.append("")
    lines = _build_category_section(all_metrics, q_meta, lines)
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- 七、逐题详细结果 ----
    lines.append("## 七、逐题详细结果")
    lines.append("")

    if has_aq:
        lines.append("| # | 问题 | 类别 | L1 R@50 | L1 MRR | Fact Cov | Cit Rec | ROUGE-L | Hallu |")
        lines.append("|---|------|------|---------|--------|----------|---------|---------|-------|")
        for m, aq in zip(all_metrics, aq_list):
            cat = q_meta.get(m.question_id, {}).get("category", "-")
            lines.append(
                f"| Q{m.question_id} | {m.question[:35]}... | {cat} | "
                f"{m.layer1.recall_at_50:.2f} | {m.layer1.mrr:.3f} | "
                f"{aq.fact_coverage:.2f} | {aq.citation_recall:.2f} | "
                f"{aq.rouge_l_f1:.2f} | {aq.hallucination_count} |"
            )
    else:
        lines.append("| # | 问题 | 类别 | L1 Recall@50 | L1 MRR | L2 Doc | L3 MRR Δ | KW Hit |")
        lines.append("|---|------|------|-------------|--------|--------|----------|--------|")
        for m in all_metrics:
            cat = q_meta.get(m.question_id, {}).get("category", "-")
            lines.append(
                f"| Q{m.question_id} | {m.question[:40]}... | {cat} | "
                f"{m.layer1.recall_at_50:.3f} | {m.layer1.mrr:.3f} | "
                f"{m.layer2.doc_recall:.3f} | {m.layer3.mrr_delta:+.3f} | "
                f"{m.keyword_hit_rate:.2f} |"
            )
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- 八、总结 ----
    lines.append("## 八、总结与建议")
    lines.append("")

    # 自动生成一些建议
    if l1.get("avg_recall_at_50", 0) < 0.5:
        lines.append("- ⚠️ Recall@50 偏低，建议检查混合检索的 dense/sparse 权重配置")
    if l3.get("total_degradation_cases", 0) > len(all_metrics) * 0.1:
        lines.append("- ⚠️ 重排退化比例偏高，建议检查 Reranker 元数据 boost 配置")
    if l1.get("avg_mrr", 0) < 0.3:
        lines.append("- ⚠️ MRR 偏低，正确答案排名靠后，建议优化粗排质量")
    if l3.get("avg_mrr_delta", 0) > 0.05:
        lines.append("- ✅ Reranker 显著改善了排序质量")
    if not degradation_cases:
        lines.append("- ✅ 无退化案例，Reranker 表现良好")
    if l2.get("domain_accuracy", 0) > 0.9:
        lines.append("- ✅ Domain 过滤准确率良好")

    lines.append("")
    lines.append(f"*报告由 eval/run_eval.py 自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    return "\n".join(lines)


def _build_category_section(
    all_metrics: List[PerQuestionMetrics],
    q_meta: Dict,
    lines: List[str],
) -> List[str]:
    """构建分类别明细表，带进度条。"""
    # 按 category 分组
    groups = defaultdict(list)
    for m in all_metrics:
        cat = q_meta.get(m.question_id, {}).get("category", "未知")
        groups[cat].append(m)

    lines.append("| 类别 | 题目数 | L1 Recall@50 | L1 MRR | L3 MRR Δ | KW Hit | 评级 |")
    lines.append("|------|--------|-------------|--------|----------|--------|------|")

    for cat in sorted(groups.keys()):
        metrics_list = groups[cat]
        n = len(metrics_list)
        avg_recall = sum(m.layer1.recall_at_50 for m in metrics_list) / n
        avg_mrr = sum(m.layer1.mrr for m in metrics_list) / n
        avg_mrr_delta = sum(m.layer3.mrr_delta for m in metrics_list) / n
        avg_kw = sum(m.keyword_hit_rate for m in metrics_list) / n
        stars = _stars(avg_kw * 100)  # 基于关键词命中率给星

        bar = _progress_bar(avg_kw * 100, width=16)
        lines.append(
            f"| {cat} | {n} | {avg_recall:.3f} | {avg_mrr:.3f} | "
            f"{avg_mrr_delta:+.3f} | {avg_kw:.0%} {bar} | {stars} |"
        )

    return lines


# ============================================================
# JSON 生成
# ============================================================

def _build_json(
    all_metrics: List[PerQuestionMetrics],
    q_meta: Dict,
    dataset: Dict,
    timestamp: str,
    answer_quality_list: Optional[List[AnswerQualityMetrics]] = None,
) -> Dict:
    agg = aggregate_metrics(all_metrics)

    # 按类别分组
    by_category = defaultdict(lambda: {
        "total": 0,
        "avg_recall_at_50": 0.0,
        "avg_mrr": 0.0,
        "avg_kw_hit_rate": 0.0,
    })
    for m in all_metrics:
        cat = q_meta.get(m.question_id, {}).get("category", "未知")
        by_category[cat]["total"] += 1
    for cat, info in by_category.items():
        cat_metrics = [m for m in all_metrics if q_meta.get(m.question_id, {}).get("category", "未知") == cat]
        n = info["total"]
        info["avg_recall_at_50"] = sum(m.layer1.recall_at_50 for m in cat_metrics) / n
        info["avg_mrr"] = sum(m.layer1.mrr for m in cat_metrics) / n
        info["avg_kw_hit_rate"] = sum(m.keyword_hit_rate for m in cat_metrics) / n

    per_question = []
    aq_iter = answer_quality_list or []

    for i, m in enumerate(all_metrics):
        entry = {
            "id": m.question_id,
            "question": m.question,
            "category": q_meta.get(m.question_id, {}).get("category", ""),
            "expected_keywords": m.expected_keywords,
            "expected_top1_doc": m.expected_top1_doc,
            "layer1": {
                "recall_at_10": m.layer1.recall_at_10,
                "recall_at_30": m.layer1.recall_at_30,
                "recall_at_50": m.layer1.recall_at_50,
                "precision_at_5": m.layer1.precision_at_5,
                "precision_at_10": m.layer1.precision_at_10,
                "mrr": m.layer1.mrr,
                "ndcg_at_10": m.layer1.ndcg_at_10,
                "hit_rate": m.layer1.hit_rate,
            },
            "layer2": {
                "doc_recall": m.layer2.doc_recall,
                "chunk_recall": m.layer2.chunk_recall,
                "cross_doc_coverage_pct": m.layer2.cross_doc_coverage_pct,
                "domain_accuracy": m.layer2.domain_accuracy,
                "category_accuracy": m.layer2.category_accuracy,
            },
            "layer3": {
                "top1_improvement": m.layer3.top1_improvement,
                "mrr_delta": m.layer3.mrr_delta,
                "ndcg_delta": m.layer3.ndcg_delta,
                "degradation": bool(m.layer3.degradation_count),
                "coarse_top1_rank": m.layer3.coarse_top1_rank,
                "fine_top1_rank": m.layer3.fine_top1_rank,
            },
            "keyword": {
                "hits": m.keyword_hits,
                "hit_rate": m.keyword_hit_rate,
            },
            "elapsed_ms": m.elapsed_ms,
            "total_candidates": m.total_candidates,
            "relevant_chunks_found": len(m.relevant_ids_used),
        }
        # 附加答案质量
        if i < len(aq_iter):
            aq = aq_iter[i]
            entry["answer_quality"] = {
                "fact_coverage": aq.fact_coverage,
                "fact_hit": aq.fact_hit_count,
                "fact_partial": aq.fact_partial_count,
                "fact_miss": aq.fact_miss_count,
                "citation_precision": aq.citation_precision,
                "citation_recall": aq.citation_recall,
                "rouge_l_f1": aq.rouge_l_f1,
                "rouge_l_precision": aq.rouge_l_precision,
                "rouge_l_recall": aq.rouge_l_recall,
                "answer_length": aq.answer_length,
                "gold_length": aq.gold_length,
                "length_ratio": aq.length_ratio,
                "hallucination_count": aq.hallucination_count,
                "hallucination_details": aq.hallucination_details,
            }
        per_question.append(entry)

    result = {
        "dataset_name": dataset.get("dataset_name", ""),
        "evaluation_date": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "coarse_top_k": 50,
            "fine_top_k": 15,
        },
        "aggregate_metrics": agg,
        "by_category": {cat: info for cat, info in by_category.items()},
        "per_question": per_question,
    }

    # 聚合答案质量
    if aq_iter:
        n_aq = len(aq_iter)
        result["aggregate_answer_quality"] = {
            "avg_fact_coverage": sum(a.fact_coverage for a in aq_iter) / n_aq,
            "avg_citation_precision": sum(a.citation_precision for a in aq_iter) / n_aq,
            "avg_citation_recall": sum(a.citation_recall for a in aq_iter) / n_aq,
            "avg_rouge_l_f1": sum(a.rouge_l_f1 for a in aq_iter) / n_aq,
            "avg_length_ratio": sum(a.length_ratio for a in aq_iter) / n_aq,
            "total_hallucinations": sum(a.hallucination_count for a in aq_iter),
        }

    return result


# ============================================================
# 辅助函数
# ============================================================

def _progress_bar(value: float, width: int = 20) -> str:
    """Unicode 进度条: ████████░░░░░░"""
    if value < 0:
        value = 0
    if value > 100:
        value = 100
    filled = int(round(value / 100 * width))
    empty = width - filled
    return f"`{'█' * filled}{'░' * empty}`"


def _stars(score: float) -> str:
    """分数转星级 (0-100 -> ⭐)"""
    if score >= 90:
        return "⭐⭐⭐⭐⭐"
    elif score >= 75:
        return "⭐⭐⭐⭐"
    elif score >= 60:
        return "⭐⭐⭐"
    elif score >= 40:
        return "⭐⭐"
    else:
        return "⭐"


# ============================================================
# LLM vs 金标 对比文档
# ============================================================

def generate_comparison_doc(
    llm_answers: List[Dict],
    dataset_name: str,
    output_dir: str = "eval/output",
) -> str:
    """
    生成 LLM答案 vs 金标答案 逐题对比 Markdown文档。

    Args:
        llm_answers: list of dict, each with:
            id, question, category, gold_answer, llm_answer,
            key_facts, expected_citations, citations, sources,
            fact_coverage, citation_recall, rouge_l, hallucination_count
        dataset_name: 数据集名称
        output_dir: 输出目录

    Returns:
        对比文档路径
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"answer_comparison_{timestamp}.md")

    lines = []
    lines.append(f"# LLM答案 vs 金标答案 逐题对比")
    lines.append(f"")
    lines.append(f"> 数据集: {dataset_name}")
    lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 题目数: {len(llm_answers)}")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # 按 category 分组索引
    from collections import defaultdict
    by_cat = defaultdict(list)
    for item in llm_answers:
        by_cat[item.get("category", "未知")].append(item)

    lines.append("## 分类索引")
    lines.append("")
    for cat in sorted(by_cat.keys()):
        items = by_cat[cat]
        qids = ", ".join(f"[Q{i['id']}](#q{i['id']})" for i in items)
        avg_fact = sum(i.get("fact_coverage", 0) for i in items) / len(items)
        avg_rl = sum(i.get("rouge_l", 0) for i in items) / len(items)
        lines.append(f"- **{cat}** ({len(items)}题, Fact={avg_fact:.2f}, ROUGE-L={avg_rl:.2f}): {qids}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 逐题对比
    for item in llm_answers:
        qid = item["id"]
        question = item["question"]
        category = item.get("category", "")
        gold = item.get("gold_answer", "")
        llm = item.get("llm_answer", "")
        key_facts = item.get("key_facts", [])
        exp_cit = item.get("expected_citations", [])
        gen_cit = item.get("citations", [])
        fact_cov = item.get("fact_coverage", 0.0)
        cit_rec = item.get("citation_recall", 0.0)
        rl = item.get("rouge_l", 0.0)
        hallu = item.get("hallucination_count", 0)

        lines.append(f"## Q{qid} — {category}")
        lines.append(f"")
        lines.append(f"**问题**: {question}")
        lines.append(f"")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|-----|")
        lines.append(f"| 事实覆盖度 | {fact_cov:.2f} |")
        lines.append(f"| 引用召回率 | {cit_rec:.2f} |")
        lines.append(f"| ROUGE-L F1 | {rl:.3f} |")
        lines.append(f"| 可疑引用数 | {hallu} |")
        lines.append(f"")

        # 期望引用 vs 实际引用
        lines.append(f"**期望引用**: {', '.join(exp_cit) if exp_cit else '(无)'}")
        lines.append(f"")
        lines.append(f"**实际引用**: {', '.join(gen_cit[:10]) if gen_cit else '(无)'}")
        lines.append(f"")

        # ---- 检索排名偏差分析 ----
        rank = item.get("rank_analysis", {})
        if rank:
            diag = rank.get("diagnosis", "")
            kw_best = rank.get("kw_hit_best", 0)
            kw_total = rank.get("kw_total", 0)
            coarse_r = rank.get("best_coarse_rank", -1)
            fine_r = rank.get("best_fine_rank", -1)
            best_file = rank.get("best_file", "")
            best_text = rank.get("text_best", "")

            # 根据严重程度加 emoji
            if diag.startswith("检索缺失"):
                emoji = "🔴"
            elif "挤出" in diag or "退化" in diag or "未进入" in diag:
                emoji = "🟡"
            elif "改善" in diag or "已在Top" in diag:
                emoji = "🟢"
            else:
                emoji = "⚪"

            lines.append(f"### {emoji} 检索排名分析")
            lines.append(f"")
            lines.append(f"| 项目 | 值 |")
            lines.append(f"|------|-----|")
            lines.append(f"| 诊断 | **{diag}** |")
            lines.append(f"| 最佳chunk关键词命中 | {kw_best}/{kw_total} |")
            lines.append(f"| 粗排排名 (Top-50) | {'第'+str(coarse_r)+'位' if coarse_r > 0 else '未找到'} |")
            lines.append(f"| 精排排名 (Top-10) | {'第'+str(fine_r)+'位' if fine_r > 0 else '已挤出' if coarse_r > 0 else '—'} |")
            if best_file:
                fname = best_file.split("/")[-1].split("\\")[-1][:60]
                lines.append(f"| 来源文件 | {fname} |")
            lines.append(f"")

            if best_text:
                lines.append(f"**最佳chunk文本 (粗排第{coarse_r}位, 含{kw_best}/{kw_total}关键词):**")
                lines.append(f"")
                lines.append(f"```")
                lines.append(best_text)
                lines.append(f"```")
                lines.append(f"")

        # ---- 检索 Top-N 切片 (LLM 实际看到的内容) ----
        top_chunks = item.get("top_chunks", [])
        if top_chunks:
            lines.append(f"### 检索 Top-{len(top_chunks)} 切片 (送入LLM上下文)")
            lines.append(f"")
            lines.append(f"| # | Score | Conf | 文件 | 编号 | 域 |")
            lines.append(f"|---|-------|------|------|------|-----|")
            for c in top_chunks:
                fname = c.get("file", "").split("/")[-1].split("\\")[-1][:40] or "-"
                lines.append(
                    f"| {c['rank']} | {c['score']:.4f} | {c['confidence']:.2f} | "
                    f"{fname} | {c.get('doc_number','')[:25]} | {c.get('domain','')} |"
                )
            lines.append(f"")
            # 展示每个切片的文本 (折叠)
            for c in top_chunks:
                fname = c.get("file", "").split("/")[-1].split("\\")[-1][:50] or "-"
                text = c.get("text", "")
                lines.append(f"<details>")
                lines.append(f"<summary>#{c['rank']} — {fname}</summary>")
                lines.append(f"")
                lines.append(f"```")
                lines.append(text[:800])
                lines.append(f"```")
                lines.append(f"")
                lines.append(f"</details>")
                lines.append(f"")
        else:
            lines.append(f"*(无检索结果)*")
            lines.append(f"")

        # 关键事实
        if key_facts:
            lines.append(f"**关键事实 ({len(key_facts)}条)**:")
            for j, kf in enumerate(key_facts, 1):
                lines.append(f"  {j}. {kf}")
            lines.append(f"")

        # 金标 vs LLM 并列
        lines.append(f"### 金标答案")
        lines.append(f"")
        lines.append(gold)
        lines.append(f"")
        lines.append(f"### LLM 生成答案")
        lines.append(f"")
        lines.append(llm)
        lines.append(f"")

        lines.append(f"---")
        lines.append(f"")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path
