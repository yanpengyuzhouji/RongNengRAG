#!/usr/bin/env python3
"""
RAG 端到端评估执行器 — 三层检索 + 答案质量评估

用法:
    python eval/run_eval.py                           # 默认
    python eval/run_eval.py --api-url http://localhost:8000
    python eval/run_eval.py --top-k 10

前置条件:
    - API 服务器已启动: python src/api/main.py
    - Milvus Lite 已初始化、模型已加载
"""
import sys
import os
import json
import argparse
import time
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from eval.metrics import (
    compute_relevance_ids_from_keywords,
    compute_all_metrics,
    compute_answer_quality,
    aggregate_metrics,
    PerQuestionMetrics,
    AnswerQualityMetrics,
)
from eval.report_generator import generate_report, generate_comparison_doc


# ============================================================
# 检索排名偏差分析
# ============================================================

def find_best_chunk_rank(
    keywords: List[str],
    coarse_results: List[Dict],
    fine_results: List[Dict],
) -> Dict[str, any]:
    """
    在粗排50条中搜索关键词命中率最高的chunk，记录其粗排/精排排名。

    返回:
        best_coarse_rank: 最佳chunk在粗排中的排名 (1-based), -1 表示未找到
        best_fine_rank:   最佳chunk在精排中的排名 (1-based), -1 表示被挤出
        kw_hit_best:      最佳chunk的关键词命中数
        kw_total:         总关键词数
        file_best:        最佳chunk所属文件
        text_best:        最佳chunk文本片段 (前300字)
        diagnosis:        "检索缺失" | "排名靠后未进入Top10" | "已排入Top10" | "无关键词"
    """
    if not keywords or not coarse_results:
        return {
            "best_coarse_rank": -1,
            "best_fine_rank": -1,
            "kw_hit_best": 0,
            "kw_total": len(keywords),
            "file_best": "",
            "text_best": "",
            "diagnosis": "无关键词",
        }

    # 在粗排中找关键词命中最多的chunk
    best_coarse_rank = -1
    best_kw_hit = 0
    best_chunk_id = ""
    best_file = ""
    best_text = ""

    for i, c in enumerate(coarse_results):
        text_lower = c.get("text", "").lower()
        hit = sum(1 for kw in keywords if kw.lower() in text_lower)
        if hit > best_kw_hit:
            best_kw_hit = hit
            best_coarse_rank = i + 1
            best_chunk_id = c.get("chunk_id", "")
            best_file = c.get("file_path", "")
            best_text = c.get("text", "")

    # 在精排中找这个chunk的排名
    best_fine_rank = -1
    if best_chunk_id:
        for j, f in enumerate(fine_results):
            if f.get("chunk_id", "") == best_chunk_id:
                best_fine_rank = j + 1
                break

    # 诊断
    if best_kw_hit == 0:
        diagnosis = "检索缺失: 50条候选池中无chunk包含任何预期关键词"
    elif best_coarse_rank > 10 and best_fine_rank == -1:
        diagnosis = f"粗排第{best_coarse_rank}位, 精排后被挤出Top-{len(fine_results)}"
    elif best_coarse_rank > 10 and best_fine_rank > 10:
        diagnosis = f"粗排第{best_coarse_rank}位, 精排后第{best_fine_rank}位, 未进入Top-10"
    elif best_coarse_rank > 10 and best_fine_rank <= 10:
        diagnosis = f"粗排第{best_coarse_rank}位→精排后提升到第{best_fine_rank}位 (Reranker改善)"
    elif best_coarse_rank <= 10 and best_fine_rank <= 10:
        diagnosis = f"粗排第{best_coarse_rank}位→精排后第{best_fine_rank}位 (已在Top-10内)"
    elif best_coarse_rank <= 10 and best_fine_rank == -1:
        diagnosis = f"粗排第{best_coarse_rank}位, 精排后反被挤掉 (Reranker退化)"
    else:
        diagnosis = f"粗排第{best_coarse_rank}位"

    return {
        "best_coarse_rank": best_coarse_rank,
        "best_fine_rank": best_fine_rank,
        "kw_hit_best": best_kw_hit,
        "kw_total": len(keywords),
        "file_best": best_file or "",
        "text_best": best_text[:400] if best_text else "",
        "diagnosis": diagnosis,
    }


# ============================================================
# HTTP 客户端
# ============================================================

def http_post(url: str, data: dict, timeout: int = 120) -> dict:
    """HTTP POST JSON, 返回解析后的 dict."""
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ConnectionError(f"无法连接 API: {e}\n请确认 python src/api/main.py 已启动") from e


def check_api_health(api_url: str) -> bool:
    """检查 API 是否在线."""
    try:
        req = urllib.request.Request(f"{api_url}/health", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("status") == "ok"
    except Exception:
        return False


# ============================================================
# 评测主循环
# ============================================================

def run_evaluation(
    dataset_path: str,
    api_url: str = "http://localhost:8000",
    top_k: int = 15,
    output_dir: str = "eval/output",
) -> str:
    """
    端到端评估: 调 /search 获取检索指标, 调 /ask 获取答案质量.

    Returns:
        Markdown 报告路径
    """
    # 加载数据集
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    questions = dataset["questions"]

    # 检查 API
    print(f"\n{'='*60}")
    print(f"  RAG 端到端评估框架 — 三层检索 + 答案质量")
    print(f"{'='*60}")
    print(f"  API: {api_url}")
    print(f"  数据集: {dataset['dataset_name']}")
    print(f"  题目数: {len(questions)}")
    print(f"  精排 Top-K: {top_k}")
    print(f"{'='*60}\n")

    if not check_api_health(api_url):
        print("[ERROR] API 服务不可达, 请先启动: python src/api/main.py")
        sys.exit(1)

    print("[OK] API 服务在线\n")

    all_metrics: List[PerQuestionMetrics] = []
    all_answer_quality: List[AnswerQualityMetrics] = []
    llm_answers: List[Dict] = []  # 收集 LLM 答案用于对比文档
    total_start = time.time()

    for i, q_data in enumerate(questions):
        qid = q_data["id"]
        question = q_data["question"]
        domain_filter = q_data.get("domain_filter", None)
        keywords = q_data.get("expected_keywords", [])
        category = q_data.get("category", "")
        gold_answer = q_data.get("gold_answer", "")
        key_facts = q_data.get("key_facts", [])
        expected_citations = q_data.get("expected_citations", [])

        question_short = question[:55] + "..." if len(question) > 55 else question
        print(f"[{i+1}/{len(questions)}] Q{qid} [{category}] {question_short}")

        # ---- Step 1: POST /search (带粗排快照) ----
        try:
            search_resp = http_post(f"{api_url}/search", {
                "query": question,
                "top_k": top_k,
                "domain_filter": domain_filter,
                "return_coarse_results": True,
            })
        except Exception as e:
            print(f"    [ERROR] /search 失败: {e}")
            all_metrics.append(PerQuestionMetrics(
                question_id=qid, question=question,
                expected_keywords=keywords,
                expected_top1_doc=q_data.get("expected_top1_doc", ""),
                domain_filter=domain_filter or "",
            ))
            all_answer_quality.append(AnswerQualityMetrics())
            continue

        # 构建 coarse/fine 结果
        coarse_results = []
        for c in (search_resp.get("coarse_results") or []):
            coarse_results.append({
                "chunk_id": c.get("chunk_id", ""),
                "file_path": c.get("file_path", ""),
                "score": c.get("score", 0),
                "confidence": c.get("confidence", 0),
                "text": c.get("text", ""),
                "domain": c.get("domain", ""),
                "category": c.get("category", ""),
            })

        fine_results = []
        for r in (search_resp.get("results") or []):
            fine_results.append({
                "chunk_id": r.get("chunk_id", ""),
                "file_path": r.get("file_path", ""),
                "score": r.get("score", 0),
                "confidence": r.get("confidence", 0),
                "text": r.get("text", ""),
                "domain": r.get("domain", ""),
                "category": r.get("category", ""),
                "doc_number": r.get("doc_number", ""),
                "page_num": r.get("page_num", 0),
            })

        # 推导相关性
        relevant_ids = set(q_data.get("relevant_chunks", [])) or None

        # 检索指标
        qm = compute_all_metrics(
            question_data=q_data,
            coarse_results=coarse_results,
            fine_results=fine_results,
            elapsed_ms=search_resp.get("elapsed_ms", 0),
            relevant_ids_override=relevant_ids,
        )
        all_metrics.append(qm)

        # ---- Step 2: POST /ask (端到端答案) ----
        aq_metrics = AnswerQualityMetrics()
        try:
            ask_resp = http_post(f"{api_url}/ask", {
                "query": question,
                "top_k": top_k,
                "domain_filter": domain_filter,
            })
            answer_text = ask_resp.get("answer", "")
            citations = ask_resp.get("citations", [])
            sources = ask_resp.get("sources", [])

            # 检索排名偏差分析：正确答案在粗排50条中排第几
            rank_analysis = find_best_chunk_rank(keywords, coarse_results, fine_results)

            # 收集 LLM 答案 + 检索 Top-N 切片 + 排名分析 用于对比文档
            top_chunks = []
            for j, r in enumerate(search_resp.get("results", [])[:10]):
                top_chunks.append({
                    "rank": j + 1,
                    "score": round(r.get("score", 0), 4),
                    "confidence": round(r.get("confidence", 0), 4),
                    "file": r.get("file_path", ""),
                    "doc_number": r.get("doc_number", ""),
                    "domain": r.get("domain", ""),
                    "category": r.get("category", ""),
                    "text": r.get("text", ""),
                })

            llm_answers.append({
                "id": qid,
                "question": question,
                "category": category,
                "domain_filter": domain_filter,
                "gold_answer": gold_answer,
                "llm_answer": answer_text,
                "key_facts": key_facts,
                "expected_citations": expected_citations,
                "citations": citations,
                "sources": sources,
                "top_chunks": top_chunks,
                "rank_analysis": rank_analysis,
                "fact_coverage": 0.0,
                "citation_recall": 0.0,
                "rouge_l": 0.0,
                "hallucination_count": 0,
            })

            if gold_answer:
                aq_metrics = compute_answer_quality(
                    answer=answer_text,
                    gold_answer=gold_answer,
                    key_facts=key_facts,
                    expected_citations=expected_citations,
                    sources=sources,
                )
                # 更新对比数据
                llm_answers[-1]["fact_coverage"] = aq_metrics.fact_coverage
                llm_answers[-1]["citation_recall"] = aq_metrics.citation_recall
                llm_answers[-1]["rouge_l"] = aq_metrics.rouge_l_f1
                llm_answers[-1]["hallucination_count"] = aq_metrics.hallucination_count
        except Exception as e:
            print(f"    [ERROR] /ask 失败: {e}")

        all_answer_quality.append(aq_metrics)

        # 一行摘要
        l1_str = f"R@50={qm.layer1.recall_at_50:.2f} MRR={qm.layer1.mrr:.3f}"
        l3_str = f"MRRΔ={qm.layer3.mrr_delta:+.3f}"
        aq_str = f"Fact={aq_metrics.fact_coverage:.2f} CitR={aq_metrics.citation_recall:.2f} R-L={aq_metrics.rouge_l_f1:.2f}"
        hallu_str = f" Hallu={aq_metrics.hallucination_count}" if aq_metrics.hallucination_count else ""
        print(f"    L1 {l1_str} | L3 {l3_str} | AQ {aq_str}{hallu_str}")

    total_elapsed = (time.time() - total_start) * 1000
    avg_elapsed = total_elapsed / len(questions) if questions else 0

    print(f"\n{'='*60}")
    print(f"  评估完成 | 总耗时: {total_elapsed:.0f}ms | 平均: {avg_elapsed:.0f}ms/题")
    print(f"{'='*60}\n")

    # ---- 生成报告 ----
    print("[report] 正在生成报告...")
    md_path, json_path = generate_report(
        all_metrics=all_metrics,
        dataset=dataset,
        answer_quality_list=all_answer_quality,
        output_dir=output_dir,
    )

    # 生成 LLM vs 金标对比文档
    cmp_path = generate_comparison_doc(
        llm_answers=llm_answers,
        dataset_name=dataset.get("dataset_name", ""),
        output_dir=output_dir,
    )

    print(f"\n  报告已生成:")
    print(f"    Markdown: {md_path}")
    print(f"    JSON:     {json_path}")
    print(f"    对比文档:  {cmp_path}")

    # 打印答案质量摘要
    if all_answer_quality:
        avg_fact = sum(a.fact_coverage for a in all_answer_quality) / len(all_answer_quality)
        avg_cit_r = sum(a.citation_recall for a in all_answer_quality) / len(all_answer_quality)
        avg_rouge = sum(a.rouge_l_f1 for a in all_answer_quality) / len(all_answer_quality)
        total_hallu = sum(a.hallucination_count for a in all_answer_quality)
        print(f"\n  答案质量摘要:")
        print(f"    事实覆盖度: {avg_fact:.3f}")
        print(f"    引用召回率: {avg_cit_r:.3f}")
        print(f"    ROUGE-L F1: {avg_rouge:.3f}")
        print(f"    可疑引用数: {total_hallu}")

    return md_path


def main():
    parser = argparse.ArgumentParser(description="RAG 端到端评估 — 三层检索 + 答案质量")
    parser.add_argument(
        "--dataset",
        default=os.path.join(PROJECT_ROOT, "eval", "datasets", "songdianshusong_biaozhunguifan.json"),
        help="评估数据集路径"
    )
    parser.add_argument(
        "--api-url", default="http://localhost:8000",
        help="API 服务器地址 (默认: http://localhost:8000)"
    )
    parser.add_argument(
        "--top-k", type=int, default=15,
        help="最终返回结果数 (默认: 15)"
    )
    parser.add_argument(
        "--output", default=os.path.join(PROJECT_ROOT, "eval", "output"),
        help="输出目录 (默认: eval/output)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"[ERROR] 数据集不存在: {args.dataset}")
        print("请先运行: python eval/build_dataset.py")
        sys.exit(1)

    run_evaluation(
        dataset_path=args.dataset,
        api_url=args.api_url,
        top_k=args.top_k,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
