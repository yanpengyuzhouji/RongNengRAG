"""
一次性工具: 从 _test_results.json 提取标注数据集
输出为干净的 eval/datasets/ JSON 格式

运行: python eval/extract_dataset.py
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def extract_dataset(source_path: str, output_path: str):
    """从 _test_results.json 提取30题到干净数据集格式。"""

    with open(source_path, "r", encoding="utf-8") as f:
        source = json.load(f)

    questions = []
    for item in source["detailed_results"]:
        q = {
            "id": item["id"],
            "question": item["question"],
            "domain_filter": source.get("domain", "送电输电"),
            "category": item["category"],
            "source_doc": item["source_doc"],
            "expected_keywords": item["expected_keywords"],
            "relevant_chunks": [],  # v2 手动标注
            "expected_top1_doc": item.get("top_file", ""),
        }
        questions.append(q)

    dataset = {
        "dataset_name": f"{source.get('domain', '')}/{source.get('category', '')} 检索评估数据集",
        "domain": source.get("domain", ""),
        "category": source.get("category", ""),
        "version": "1.0",
        "created": "2026-06-12",
        "source": f"extracted from {os.path.basename(source_path)}",
        "test_date": source.get("test_date", ""),
        "total_questions": len(questions),
        "questions": questions,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"[OK] 从 {source_path} 提取了 {len(questions)} 道题目 -> {output_path}")

    # 打印分类统计
    from collections import Counter
    cat_counts = Counter(q["category"] for q in questions)
    print("\n分类统计:")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count} 题")


if __name__ == "__main__":
    project_root = os.path.join(os.path.dirname(__file__), "..")
    source = os.path.join(project_root, "_test_results.json")
    output = os.path.join(project_root, "eval", "datasets", "songdianshusong_biaozhunguifan.json")

    if not os.path.exists(source):
        print(f"[ERROR] 源文件不存在: {source}")
        sys.exit(1)

    extract_dataset(source, output)
