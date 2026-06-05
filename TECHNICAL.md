# 榕能电力审图知识库 RAG — 技术文档

## 1. 系统架构

### 1.1 整体架构

系统采用前后端分离 + HTTP API 架构：

```
浏览器 (Gradio UI :7860)
    │  HTTP/1.1 JSON
    ▼
FastAPI (:8000)
    │
    ├── FileProcessor ──→ Chunker + Embedder + MilvusStore
    ├── Retriever     ──→ QueryAnalyzer + Embedder + MilvusStore + Reranker
    └── LLMEngine     ──→ Ollama API (:11434)
```

### 1.2 关键设计决策

| 决策 | 原因 |
|------|------|
| Gradio 作为 HTTP 客户端 | Milvus Lite 仅支持单进程，Gradio 不直接操作数据库 |
| Ollama 本地模型 | 无需外网，数据不出内网，CUDA 加速 |
| 嵌入式向量库 (Milvus Lite) | 无需 Docker，零配置部署 |
| 三阶段检索 (分析→召回→精排) | 兼顾召回率和准确率 |
| 延迟加载模型 | 减少冷启动内存占用 |

## 2. 模块详解

### 2.1 嵌入引擎 (`ingestion/embedder.py`)

**类**: `Embedder`

支持两种后端：

| 后端 | 模式 | 使用场景 |
|------|------|----------|
| Ollama | `provider: "ollama"` | 默认，调用 `/api/embed` 批量嵌入 |
| Sentence Transformers | `provider: "sentence_transformers"` | HuggingFace 直接加载（需联网） |

**API**:

```python
embedder = Embedder()

# 批量嵌入
result = embedder.encode(texts, show_progress=True)
# result.dense_vectors: List[List[float]]  稠密向量 (1024 维)
# result.sparse_vectors: List[dict]         稀疏向量 (Ollama 返回空)

# 单查询嵌入
dense_vec, sparse_vec = embedder.encode_query("变电消防要求")
```

**模型**: `bge-m3:latest` (1024 维稠密向量)，运行在 GPU (RTX 4070 SUPER) 上。

### 2.2 向量数据库 (`ingestion/milvus_store.py`)

**类**: `MilvusStore`

**Schema** (集合名: `power_design_chunks`):

| 字段 | 类型 | 说明 |
|------|------|------|
| chunk_id | VARCHAR(256) | 主键: `{file_hash}_{chunk_index}` |
| text | VARCHAR(65535) | 文本内容 |
| dense_vector | FLOAT_VECTOR(1024) | 稠密向量，COSINE 相似度 |
| sparse_vector | SPARSE_FLOAT_VECTOR | 稀疏词向量 |
| domain | VARCHAR(32) | 专业域 (变电/配电/送电输电/综合) |
| category | VARCHAR(64) | 文档类目 |
| publish_level | VARCHAR(32) | 发布层级 (国标/行标/企标...) |
| voltage_level | VARCHAR(32) | 电压等级 |
| discipline | VARCHAR(64) | 专业类型 (电气一次/二次/土建...) |
| equipment_type | VARCHAR(64) | 设备类型 |
| year | INT16 | 年份 |
| file_path | VARCHAR(1024) | 源文件路径 |
| doc_number | VARCHAR(256) | 文档编号 |
| page_num | INT16 | 页码 |

**重要兼容性修复**:
- `os.rename` → `os.replace` 补丁: 解决 milvus-lite 3.0 Windows `WinError 183` 问题
- `LOCK` 文件自动清理: 解决进程异常退出后的残留锁
- 集合自动创建: `_ensure_collection()` 在 insert 和 search 时自动调用
- 集合自动加载: `load_collection()` 防止 `released` 状态

### 2.3 检索器 (`retrieval/retriever.py`)

**类**: `Retriever` — 三阶段检索编排器

```
search(query, top_k, domain_filter)
    │
    ├─ 阶段0: QueryAnalyzer.analyze(query)
    │   返回 AnalyzedQuery (域/电压/设备/查询类型/扩展词/过滤表达式)
    │
    ├─ 阶段1: Embedder.encode_query() → MilvusStore.hybrid_search()
    │   RRF 融合稠密+稀疏搜索，返回 coarse_top_k=50 候选
    │
    └─ 阶段2: Reranker.rerank(candidates, query, top_k)
        嵌入相似度或交叉编码器评分 + 元数据加权
        返回最终 top_k 结果
```

**检索配置** (`config.yaml`):

```yaml
retrieval:
  coarse_top_k: 50      # 粗召回候选数
  fine_top_k: 15         # 精排返回数
  rrf_k: 60              # RRF 融合参数
  dense_weight: 0.6      # 稠密向量权重
  sparse_weight: 0.4     # 稀疏向量权重
```

### 2.4 查询分析器 (`retrieval/query_analyzer.py`)

**类**: `QueryAnalyzer`

6 步分析管道:
1. **文档编号提取**: 正则匹配 闽电/榕电/基建/运检 等发文编号
2. **域分类**: 基于关键词典打分 (变电/配电/送电输电/综合)
3. **结构化参数提取**: 电压等级、专业类型、设备类型、发布层级、年份
4. **查询类型判断**: 技术问答/文档查找/跨域对比/数值查规/通用
5. **同义词扩展**: 电力领域同义词词典 (消防→防火/灭火, 接地→接地装置...)
6. **Milvus 过滤表达式**: 构建标量过滤条件

### 2.5 重排序器 (`retrieval/reranker.py`)

**类**: `Reranker`

两种模式:

| 模式 | 算法 | 配置 |
|------|------|------|
| Ollama | 用 BGE-M3 嵌入 query + 候选文本 → 余弦相似度 | `provider: "ollama"` |
| FlagEmbedding | BGE-Reranker-v2-m3 交叉编码器 | `provider: "flagembedding"` |

**元数据加权** (Ollama 和 FlagEmbedding 通用):
- 文档编号精确匹配: ×1.5
- 标准规范类目: ×1.2
- 国标/行标: ×1.1
- 域精确匹配: ×1.1
- 电压等级匹配: ×1.15
- 图纸 (非查图查询): ×0.85

### 2.6 文件处理器 (`ingestion/file_processor.py`)

**类**: `FileProcessor` — 完整入库管道

```
process(file_path, domain, category)
    │
    ├─ Step 1: 解析 (parse_time_ms)
    │   _build_file_meta() → _parse_file()
    │   ├─ PDF: PDFParser → Chunker
    │   ├─ DOCX: python-docx → Chunker
    │   ├─ XLSX: openpyxl → Chunker
    │   └─ TXT/MD: 直接读取 → Chunker
    │
    ├─ Step 2: 嵌入 (embed_time_ms)
    │   Embedder.encode(embedding_texts)
    │
    └─ Step 3: 入库
        MilvusStore.insert(chunks, dense_vecs, sparse_vecs, emb_texts)
```

**去重**: 基于 SHA-256 文件哈希，`file_metadata.db` 中记录状态，已入库文件自动跳过。

### 2.7 LLM 推理引擎 (`generation/llm_engine.py`)

**类**: `LLMEngine`

**后端**:
- Ollama: 通过 OpenAI 兼容 API (`/v1/chat/completions`)
- llama-cpp-python: 直接加载 GGUF 模型文件

**提示词模板** (`generation/prompt_templates.py`):
- `domain_technical`: 专业问答 (强调引用来源、区分标准层级、标注强制性)
- `document_lookup`: 结构化文档摘要
- `cross_domain_comparison`: 跨域对比分析 (表格对比)
- `specification_lookup`: 精确数值查规
- `general_qa`: 通用问答

### 2.8 Gradio UI (`ui/app.py`)

HTTP 客户端模式，所有操作通过 FastAPI 完成:

| UI 功能 | API 调用 |
|---------|----------|
| 智能问答 | `POST /ask` |
| 文档检索 | `POST /search` |
| 上传入库 | `POST /upload` |
| 文件管理 | `GET /files` + `DELETE /files/{id}` |
| 入库统计 | `GET /files/summary` |
| 选择性删除 | 下拉选择文件 → `DELETE /files/{id}` → 自动刷新列表 |

## 3. API 接口文档

### POST /upload

```json
// Request: multipart/form-data
{
  "file": <binary>,
  "domain": "变电" | "配电" | "送电输电" | "综合" | null,  // 可选
  "category": "标准规范" | ... | null                       // 可选
}

// Response:
{
  "success": true,
  "file_name": "GB_50150-2016_电气装置安装工程电气设备交接试验标准.pdf",
  "file_hash": "7cf01d1c583ff4812f...",
  "status": "completed",
  "chunks_created": 67,
  "chars_extracted": 125430,
  "domain": "变电",
  "category": "标准规范",
  "doc_number": "GB 50150-2016",
  "parse_time_ms": 141.0,
  "embed_time_ms": 6828.9,
  "total_time_ms": 8153.1,
  "error_message": ""
}
```

### POST /search

```json
// Request:
{
  "query": "变电消防设计要求",
  "top_k": 15,
  "domain_filter": "变电" | null
}

// Response:
{
  "query": "变电消防设计要求",
  "query_type": "domain_technical",
  "domain": "变电",
  "total_candidates": 50,
  "elapsed_ms": 6250.0,
  "filter_applied": "domain == \"变电\"",
  "results": [
    {
      "rank": 1,
      "chunk_id": "7cf01d1c..._1",
      "text": "文本预览...",
      "score": 0.95,
      "domain": "变电",
      "category": "标准规范",
      "file_path": "D:/知识库/变电/标准规范/GB_50060.pdf",
      "doc_number": "GB 50060-2008",
      "voltage_level": "220kV",
      "publish_level": "国标",
      "page_num": 15
    }
  ]
}
```

### POST /ask

```json
// Request:
{
  "query": "变电消防设计要求有哪些？",
  "top_k": 15,
  "domain_filter": null
}

// Response:
{
  "query": "变电消防设计要求有哪些？",
  "answer": "根据检索到的高级标准规范，变电消防设计要求主要包括...【GB 50060-2008 第X条】",
  "citations": ["GB 50060-2008 第X条", "DL/T 5352-2018"],
  "sources": [
    {"file_path": "...", "doc_number": "GB 50060-2008", "domain": "变电", "category": "标准规范"}
  ],
  "elapsed_ms": 28519.0
}
```

## 4. 数据流

### 4.1 入库流程

```
用户上传文件
    │
    ▼
FastAPI /upload 接收 → 保存到 uploads/
    │
    ▼
FileProcessor.process(file_path)
    │
    ├─ compute_hash(file) → SHA-256
    ├─ 检查 file_metadata.db → 已入库? → 跳过
    ├─ _build_file_meta() → 元数据 (域/类目/编号/电压/设备...)
    ├─ _parse_file()
    │   ├─ PDF → PDFParser.parse() → 逐页文本
    │   ├─ DOCX → python-docx → 段落文本
    │   ├─ XLSX → openpyxl → 表格文本
    │   └─ Chunker → 语义分块 / 单页分块 / 全文分块
    │
    ├─ Embedder.encode(texts) → 稠密向量 + 稀疏向量
    │
    ├─ MilvusStore.insert(chunks, vectors)
    │   └─ MilvusClient.insert("power_design_chunks", rows)
    │
    └─ file_metadata.db 记录 completed 状态
```

### 4.2 检索流程

```
用户输入查询
    │
    ▼
Retriever.search(query, top_k)
    │
    ├─ QueryAnalyzer.analyze(query)
    │   ├─ 提取文档编号 → 如有，直接过滤查找
    │   ├─ 域分类 → 关键词匹配打分
    │   ├─ 提取电压/设备/专业参数
    │   └─ 同义词扩展 "消防" → "防火/灭火/火灾报警"
    │
    ├─ Embedder.encode_query(expanded_query)
    │   └─ Ollama /api/embeddings → [1024 维向量]
    │
    ├─ MilvusStore.hybrid_search(dense_vec, sparse_vec, filter_expr)
    │   ├─ 稠密搜索 (COSINE) → dense_req
    │   ├─ 稀疏搜索 (IP)    → sparse_req
    │   └─ RRF 融合 → 50 candidates
    │
    ├─ Reranker.rerank(query, candidates, top_k)
    │   ├─ Ollama 嵌入检索 query + 候选文本
    │   ├─ 余弦相似度计算
    │   ├─ 元数据加权
    │   └─ 排序取 Top-K
    │
    └─ 返回 SearchResponse {results, query_type, domain, elapsed_ms}
```

### 4.3 问答流程

```
用户输入问题
    │
    ▼
/ask 端点
    │
    ├─ Retriever.search() → 检索结果
    ├─ Retriever.format_context_for_llm() → 格式化上下文
    │
    ├─ LLMEngine.generate_rag_answer(query, context, query_type)
    │   ├─ get_prompt(query_type) → 选择合适的提示词模板
    │   ├─ 填充 {context} 和 {query}
    │   └─ Ollama /v1/chat/completions → 生成回答
    │
    └─ 返回 AskResponse {answer, citations, sources, elapsed_ms}
```

## 5. 部署配置

### 5.1 配置文件路径

所有模块默认从 `D:/rag-system/config.yaml` 读取配置。运行前需将项目根目录的 `config.yaml` 复制到该路径。

### 5.2 模型运行位置

| 模型 | 运行位置 | 方式 |
|------|----------|------|
| bge-m3:latest | GPU | Ollama 自动分配到 CUDA |
| qwen3:8b | GPU | Ollama 自动分配到 CUDA |

### 5.3 数据存储

| 路径 | 内容 |
|------|------|
| `D:/rag-system/data/milvus_lite.db/` | Milvus Lite 向量数据 (LMDB) |
| `D:/rag-system/data/file_metadata.db` | SQLite 文件注册表 |
| `D:/rag-system/data/uploads/` | 上传文件暂存 |
| `D:/rag-system/data/parsed_cache/` | 解析缓存 |
| `D:/rag-system/config.yaml` | 运行时配置 |

### 5.4 从 Milvus Lite 迁移到 Milvus Standalone

当需要多进程共享或更高并发时:

```python
# 当前 (Lite)
client = MilvusClient("D:/rag-system/data/milvus_lite.db")

# 迁移到 Docker Standalone
client = MilvusClient(uri="http://localhost:19530")
```

Schema 和 API 完全兼容，无需修改业务代码。

## 6. 性能参数

| 参数 | 值 |
|------|-----|
| 嵌入维度 | 1024 (BGE-M3) |
| 单次嵌入耗时 | ~7s / 16 chunks (GPU) |
| 检索耗时 | ~6s (含嵌入+搜索+重排) |
| 问答总耗时 | ~28s (含检索+LLM生成) |
| 单文件入库 | ~8s (含解析+嵌入+入库) |
| 粗召回候选数 | 50 |
| 精排结果数 | 15 (可配置) |
