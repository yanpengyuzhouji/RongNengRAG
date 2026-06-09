# 榕能电力审图知识库 RAG — 技术文档

## 1. 系统架构

### 1.1 整体架构

系统采用前后端分离 + SSE 流式 + 多轮对话架构：

```
浏览器 (Gradio UI :7860)
    │  SSE Stream + HTTP JSON
    ▼
FastAPI (:8000)
    │
    ├── FileProcessor ──→ Chunker + Embedder + MilvusStore
    ├── Retriever     ──→ QueryAnalyzer + Embedder + MilvusStore + Reranker
    ├── LLMEngine     ──→ Provider (Ollama/Bailian)
    │                      └── Ollama /api/chat (:11434)
    └── ConversationManager ──→ 多轮对话 + 上下文压缩
```

### 1.2 关键设计决策

| 决策 | 原因 |
|------|------|
| Gradio 作为 HTTP 客户端 | Milvus Lite 仅支持单进程，Gradio 不直接操作数据库 |
| Ollama 原生 `/api/chat` | OpenAI 兼容接口丢弃 Qwen3 thinking 字段 |
| Provider 模式 | Ollama/百炼/llama-cpp 后端解耦，一键切换 |
| Qwen3.5 + think:false | 推理模型思维链占 85% token，关闭后提速 10 倍 |
| SSE 流式生成 | token 级实时推送 + 心跳防超时 |
| 会话侧边栏 + 隔离 | 每会话独立历史，自动压缩超窗口消息 |
| BGE-M3 稠密+稀疏混合检索 | 原生稀疏向量，解决关键词匹配差的问题 |
| 三阶段检索 (分析→召回→精排) | 兼顾召回率和准确率 |
| 文件注册表识别 + 完整文档注入 | query 含文件名时注入完整文档，解决 chunk 检索遗漏问题 |
| 同系列文件排除 | 匹配会议材料之一时排除 02-07，防止 LLM 混淆 |
| 延迟加载 + 启动预热 | 减少冷启动内存，启动时预加载嵌入模型 |

## 2. 模块详解

### 2.1 嵌入引擎 (`ingestion/embedder.py`)

**类**: `Embedder`

支持两种后端：

| 后端 | 模式 | 使用场景 |
|------|------|----------|
| Sentence Transformers | `provider: "sentence_transformers"` | 默认，本地 GPU 加载 BGE-M3 |
| Ollama | `provider: "ollama"` | 调用 `/api/embed` 批量嵌入 |

**API**:

```python
embedder = Embedder()

# 批量嵌入 (稠密 + 稀疏)
result = embedder.encode(texts, show_progress=True)
# result.dense_vectors: List[List[float]]  稠密向量 (1024 维)
# result.sparse_vectors: List[dict]         BGE-M3 原生稀疏向量

# 单查询嵌入
dense_vec, sparse_vec = embedder.encode_query("变电消防要求")
```

**模型**: `BAAI/bge-m3` (1024 维稠密 + 原生稀疏)，运行在 GPU (RTX 4070 SUPER) 上。

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

### 2.5 文件注册表 (`retrieval/file_registry.py`)

**类**: `FileRegistry` — query 中文件名检测 + 完整文档注入的枢纽

从 SQLite `file_registry` 表读取已入库文件元数据，通过多策略检测用户 query 中是否引用了特定文件。

**四阶段检测策略** (`detect_files_in_query`):

| 阶段 | 策略 | 示例匹配 |
|------|------|----------|
| 1 | 正则提取文件名候选（书名号、引号、扩展名） | `"GB50060-2008.pdf"` → `GB50060-2008.pdf` |
| 2 | 双向子串匹配（文件名 in query / query in 文件名） | `会议材料一` → `01会议材料之一2022年...` |
| 3 | 文档编号匹配（标准编号 + 发文编号） | `闽电〔2015〕241号` → 精确文件 |
| 4 | "会议材料之X" 系列匹配 + 中文序数归一化 | `材料三` → `03会议材料之三...` |

**中文数字归一化**:
```
一→1 二→2 三→3 四→4 五→5 六→6 七→7 八→8 九→9 十→10
十一→11 ... 二十→20
```
支持 `zfill(2)` 对齐文件名中的 "01" 格式。

**五种自然语言变体**（策略4）:
- `01会议材料之一` (完整，含数字前缀+之)
- `会议材料之一` (无数字前缀，有之)
- `01会议材料一` (有数字前缀，无之)
- `会议材料一` (无前缀，无之 — 最常见)
- `材料一` / `材料三` (仅材料+数字)

**API**:
```python
registry = FileRegistry()

# 检测 query 中的文件名引用
matches = registry.detect_files_in_query("会议材料一总结")
# → [FileMatchResult(entry=..., match_type="partial", match_score=0.85)]

# 按文件名搜索
entries = registry.search_by_filename("消防", exact=False)

# 获取注册表摘要
count = registry.get_file_count()  # → 32
```

### 2.6 重排序器 (`retrieval/reranker.py`)

**类**: `Reranker`

两种模式:

| 模式 | 算法 | 配置 |
|------|------|------|
| FlagEmbedding | BGE-Reranker-v2-m3 交叉编码器 | `provider: "flagembedding"` |
| Ollama | 用 BGE-M3 嵌入 query + 候选文本 → 余弦相似度 | `provider: "ollama"` |

**元数据加权** (两种模式通用):
- 文件名匹配: ×1.30
- 文档编号精确匹配: ×1.05
- 标准规范类目: ×1.05
- 国标/行标: ×1.05
- 域精确匹配: ×1.05
- 电压等级匹配: ×1.10
- 图纸 (非查图查询): ×0.85

**置信度校准**: 每个结果附带 `confidence` (0-1)，基于 rerank 分数 min-max 归一化。

### 2.7 文件处理器 (`ingestion/file_processor.py`)

**类**: `FileProcessor` — 完整入库管道

```
process(file_path, domain, category)
    │
    ├─ Step 1: 解析 (parse_time_ms)
    │   _build_file_meta() → _parse_file()
    │   ├─ PDF: PDFParser → Chunker
    │   ├─ DOC: Word COM (win32com) → LibreOffice → olefile → antiword
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

**.doc 解析**: 通过 `win32com.client.Dispatch("Word.Application")` 自动化本机 Word（Windows），大幅提升旧版 .doc 文件解析成功率。

### 2.8 LLM 推理引擎 (`generation/llm_engine.py`)

**类**: `LLMEngine` — Provider 模式 facade

**后端**:
| Provider | 接口 | 说明 |
|----------|------|------|
| Ollama | 原生 `/api/chat` | 默认，支持 think:false 关闭思维链 |
| Bailian | 阿里云百炼 DashScope | OpenAI 兼容，需 `DASHSCOPE_API_KEY` |
| llama-cpp | 直接加载 GGUF | 备选方案 |

**Provider 架构** (`generation/providers/`):
```
LLMEngine (facade)
   └── BaseProvider (抽象)
        ├── OllamaProvider  → Ollama /api/chat
        └── BailianProvider → DashScope /v1/chat/completions
```

**提示词模板** (`generation/prompt_templates.py`):
- `specification_lookup`: 精确数值查规
- `domain_technical`: 专业问答 (强调引用来源、区分标准层级、标注强制性)
- `document_lookup`: 结构化文档摘要
- `cross_domain_comparison`: 跨域对比分析
- `general_qa`: 通用问答
- `SYSTEM_PROMPT_CHAT`: 多轮对话系统提示 (直接回答，不输出思考过程)

### 2.9 对话管理 (`generation/conversation_manager.py`)

**类**: `ConversationManager`

- 内存存储 + 日后可迁移 SQLite
- 自动压缩: 超过 `max_context_tokens * 0.85` 时，旧消息合并为摘要
- 保留最近 `keep_detail_rounds=3` 轮完整内容
- 所有时间戳使用北京时间 `Asia/Shanghai`

### 2.10 Gradio UI (`ui/app.py`)

**v3.0 重构** — 聊天界面 + 会话管理:

```
┌──────────┬──────────────────────────────────┐
│ 侧边栏    │ 主区域                            │
│          │                                  │
│ [+ 新建] │  Chatbot (gr.Chatbot type=messages)│
│ ──────── │                                  │
│ 会话1    │  user: 提问...                    │
│ 会话2    │  assistant: 回答...               │
│ 会话3    │  [2026-06-08 16:30 北京时间]      │
│          │                                  │
│          │  [输入框________________] [发送]  │
└──────────┴──────────────────────────────────┘
```

**核心功能**:
- **流式问答**: SSE → `gr.Chatbot` generator，实时 token 展示
- **会话隔离**: `gr.State` 维护 `conversation_id`，每会话独立上下文
- **心跳动画**: 检索/thinking 阶段显示 "🔍 检索中..." / "💭 思考中..."
- **会话管理**: 新建/切换/删除，API CRUD
- **文件管理**: 保留原 Tab（入库/检索/统计）

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
  "domain_filter": null,
  "conversation_id": "abc123"  // 可选, 多轮对话
}

// Response:
{
  "query": "变电消防设计要求有哪些？",
  "answer": "根据检索到的高级标准规范...【GB 50060-2008 第X条】",
  "citations": ["GB 50060-2008 第X条", "DL/T 5352-2018"],
  "sources": [...],
  "elapsed_ms": 28519.0
}
```

### POST /ask/stream

SSE 流式端点，token 级实时推送:

```
data: {"status":"searching"}          // 检索中
data: {"status":"thinking"}           // 模型思考中
data: {"token":"变电","done":false}   // 流式 token
data: {"token":"","done":true,"citations":[...],"full_answer":"..."}  // 完成
```

### POST /conversations

```json
// Request:
{"title": "变电消防规范咨询"}

// Response:
{"conv_id":"abc123","title":"...","created_at":"2026-06-08T...","message_count":0}
```

### GET/DELETE /conversations/{conv_id}

获取会话历史(含完整消息列表) 或 删除会话。

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

### 4.3 问答流程 (含文件注册表注入)

```
用户输入问题
    │
    ▼
/ask 端点
    │
    ├─ Retriever.search() → 检索结果
    │
    ├─ [新增] FileRegistry.detect_files_in_query(query)
    │   └─ 四策略匹配 → FileMatchResult? (文件名 + score + 系列标识)
    │
    ├─ 分支判断:
    │   │
    │   ├─ 检测到文件名:
    │   │   ├─ MilvusStore.query_by_file_hash() → 完整文档 chunks
    │   │   ├─ _format_full_document() → 按页码排序拼接
    │   │   ├─ _extract_series_key() → 识别系列标识
    │   │   ├─ 构建上下文:
    │   │   │   ├─ [主] 完整文档 + 强化聚焦指令
    │   │   │   ├─ [补充] 非主文件 + 非同系列 + 最多2个
    │   │   │   └─ 标注: "以下仅供背景，请勿引用"
    │   │   └─ → LLM
    │   │
    │   └─ 未检测到文件名:
    │       └─ Retriever.format_context_for_llm() → 正常格式化
    │
    ├─ LLMEngine.generate_rag_answer(query, context, query_type)
    │   ├─ get_prompt(query_type) → 选择合适的提示词模板
    │   ├─ 填充 {context} 和 {query}
    │   └─ Ollama /api/chat → 生成回答
    │
    └─ 返回 AskResponse {answer, citations, sources, elapsed_ms}
```

**文件注册表注入关键规则**:
- 匹配到文件 → **优先用 file_hash 查询**（chunk_id 前缀，最可靠）
- 完整文档前置，聚焦指令强硬: "**只**基于上述完整文档内容回答"
- 同系列文件**100% 排除**: 匹配材料01 → 自动过滤02-07
- 补充检索从 5+ 个缩减至**最多 2 个**，且标注不可引用
- 检索无结果时仍尝试从注册表拉取: 纯文件名查询也能拿到答案

## 5. 部署配置

### 5.1 配置文件

`config.yaml` 位于项目根目录，所有路径相对解析。

### 5.2 模型运行位置

| 模型 | 运行位置 | 说明 |
|------|----------|------|
| BAAI/bge-m3 | GPU (sentence_transformers) | 稠密+稀疏嵌入 |
| BAAI/bge-reranker-v2-m3 | GPU (FlagEmbedding) | 交叉编码器精排 |
| qwen3.5:4b | GPU (Ollama) | LLM 问答，think=false 关闭思考链 |
| bge-m3:latest | GPU (Ollama) | Ollama 备用嵌入 |

### 5.3 数据存储

| 路径 | 内容 |
|------|------|
| `data/milvus_lite.db/` | Milvus Lite 向量数据 (LMDB) |
| `data/file_metadata.db` | SQLite 文件注册表 |
| `data/uploads/` | 上传文件暂存 |
| `data/parsed_cache/` | 解析缓存 |
| `config.yaml` | 运行时配置 |
| `.env` | API Key (不入 git) |
| `.env.example` | API Key 模板 |

## 6. 性能参数

| 参数 | 值 |
|------|-----|
| 嵌入维度 | 1024 (BGE-M3 稠密) + 原生稀疏 |
| 单次嵌入耗时 | ~7s / 16 chunks (GPU) |
| 检索耗时 | ~3s (QueryAnalyze+HybridSearch+Rerank) |
| LLM token/s | **~112 t/s** (qwen3:4b, 32k ctx, Ollama eval_rate) |
| 问答总耗时 | ~8-15s (搜索3s + LLM生成5-12s, think=false时) |
| 粗召回候选数 | 50 |
| 精排结果数 | 15 (可配置) |
| GPU | NVIDIA RTX 4070 SUPER (12GB) |

> **注**: Qwen3 推理模型强制 thinking，112 t/s 中仅 1-15% 为有效内容（剩余被内部思维链消耗）。Qwen3.5 通过 `think=false` 彻底关闭思维链后，112 t/s 全部为有效输出。
