# RAG 检索评估框架

三层评估体系，覆盖检索质量、召回质量、重排序效果。

## 快速开始

```bash
# 1. 停止 API 服务器 (Milvus Lite 单进程限制)
#    按 Ctrl+C 停止 python src/api/main.py

# 2. 提取/更新数据集 (首次或源数据变更时)
python eval/extract_dataset.py

# 3. 运行评估
python eval/run_eval.py

# 4. 查看报告
#    eval/output/eval_report_*.md   — 可读报告
#    eval/output/eval_results_*.json — 结构化数据
```

## 自定义运行

```bash
# 指定数据集
python eval/run_eval.py --dataset eval/datasets/your_dataset.json

# 调整返回数量
python eval/run_eval.py --top-k 10

# 自定义输出目录
python eval/run_eval.py --output eval/my_results
```

## 数据集格式

数据集为 JSON 文件，存放在 `eval/datasets/`：

```json
{
  "dataset_name": "域/类目 检索评估数据集",
  "domain": "变电",
  "category": "标准规范",
  "version": "1.0",
  "total_questions": 30,
  "questions": [
    {
      "id": 1,
      "question": "66kV及以下架空电力线路杆塔...？",
      "domain_filter": "送电输电",
      "category": "架空输电线路",
      "source_doc": "GB_50061-2010_...",
      "expected_keywords": ["荷载设计值", "极限状态"],
      "relevant_chunks": [],
      "expected_top1_doc": "GB_50061-2010_...2025年版.pdf"
    }
  ]
}
```

### 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `id` | ✅ | 题目序号 |
| `question` | ✅ | 自然语言问题 |
| `domain_filter` | ✅ | 检索域过滤（变电/配电/送电输电/综合） |
| `category` | ✅ | 资料类别（架空输电线路/钢结构/...） |
| `source_doc` | ✅ | 答案来源文档编号 |
| `expected_keywords` | ✅ | 预期在检索结果中出现的关键词 |
| `relevant_chunks` | | v2 手动标注相关 chunk_id 列表（为空时自动从关键词推导） |
| `expected_top1_doc` | | 预期的 Top-1 来源文件名 |

## 三层评估框架

```
问题 → domain_filter → 混合检索 → 快照候选池(L1) → Reranker → Top-N(L3)
                                  ↓
                              召回质量(L2)
```

### Layer 1: 检索质量
评估混合检索候选池 (50条) 的覆盖力和排序质量。
- **Recall@K**: 相关 chunk 在 Top-K 候选中的覆盖率
- **Precision@K**: Top-K 中相关 chunk 的精度
- **MRR**: 第一个相关 chunk 排名的倒数均值
- **NDCG@10**: 归一化折损累计增益

### Layer 2: 召回质量
评估系统从知识库中找回所有相关材料的能力。
- **文档召回率**: 预期文档在候选池中的覆盖
- **Chunk 召回率**: 相关 chunk 的覆盖比例
- **Domain/Category 准确率**: Top-1 结果的域和类目匹配

### Layer 3: 重排序效果
评估 Reranker 对排序的改善程度。
- **Top-1 提升率**: 正确答案被提到第一位的比例
- **MRR/NDCG Delta**: 正向表示改善
- **退化检测**: 重排后排名反而下降的案例

## 目录结构

```
eval/
├── README.md                    # 本文件
├── datasets/                    # 评估数据集
│   └── songdianshusong_biaozhunguifan.json
├── extract_dataset.py           # 从 _test_results.json 提取数据集
├── metrics.py                   # 指标计算 (纯函数)
├── report_generator.py          # 报告生成器
├── run_eval.py                  # 主评测入口
└── output/                      # 评测输出 (gitignore)
    ├── eval_report_*.md
    └── eval_results_*.json
```

## 注意事项

- **单进程**: Milvus Lite 仅支持单进程，评估时确保 API 服务器已停止
- **GPU 内存**: BGE-M3 (~2GB) + BGE-Reranker (~2GB)，约需 4GB VRAM
- **首题较慢**: Reranker 首次加载约 5 秒
- **关键词推导**: `relevant_chunks` 为空时，相关性由关键词在候选池文本中匹配推导（v2 将支持手动标注）
