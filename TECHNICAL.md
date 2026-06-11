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
| PaddleOCR 扫描件识别 | 子进程隔离，自动检测文字量不足的页面并 OCR (GPU/CPU 可选) |
| GPU 显存感知调度 | 入库/搜索时按需加载模型，用完即卸，LLM 优先 |
| OCR 双槽并行 + 嵌入流水线 | 2个OCR子进程 + 1个嵌入线程同时跑，时间减半 |
| 三端一致性机制 | 注册表-向量库-物理文件 同步清理，失效文件自动检测 |

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

支持两种后端：

| 后端 | 算法 | 配置 |
|------|------|------|
| FlagEmbedding | BGE-Reranker-v2-m3 交叉编码器 | `provider: "flagembedding"` |
| Ollama | 用 BGE-M3 嵌入 query + 候选文本 → 余弦相似度 | `provider: "ollama"` |

**按需加载/卸载**:
- `_ensure_loaded()`: 首次 rerank 时加载模型（~5s）
- `unload()`: 搜索完成后释放 — `del self.model` + `torch.cuda.empty_cache()`，释放 ~2GB 显存
- Retriever 在每次 `search()` 完成后自动调用 `unload()`，下次搜索按需重载

**元数据加权** (两种模式通用):
- 文件名匹配: ×1.30
- 文档编号精确匹配: ×1.05
- 标准规范类目: ×1.05
- 国标/行标: ×1.05
- 域精确匹配: ×1.05
- 电压等级匹配: ×1.10
- 图纸 (非查图查询): ×0.85

**置信度校准**: 每个结果附带 `confidence` (0-1)，基于 rerank 分数 min-max 归一化。

### 2.7 GPU 显存监控 (`utils/gpu_monitor.py`)

**`新增`** **类**: `GpuMonitor` — GPU 显存实时监控 + 背压调度

**功能**:
- `get_vram_info()`: pynvml 实时查询 总/已用/空闲显存
- `is_ollama_busy()`: 通过 `/api/ps` 检测 Ollama 是否有模型在推理
- `wait_for_vram(min_free_mb)`: 背压等待 — 显存不足时检查 Ollama 状态，LLM 繁忙则轮询等待直至释放，LLM 空闲则短等 30s 后强制执行

**显存预算 (RTX 4070 SUPER 12GB)**:
```
BGE-M3 (常驻)        ~2.0 GB    每次查询都要做 embedding
BGE-Reranker (按需)   ~2.0 GB    搜索时加载，用完释放
PaddleOCR (按需)      ~2.5 GB    入库时加载，子进程退出释放
Ollama LLM (独立)     ~4.0 GB    独立进程，不占 Python 显存
系统 + 桌面           ~1.5 GB    Windows + VS Code + Edge
────────────────────────────────
峰值并行               ~8.0 GB    12GB 容量内有 33% 裕量
常态                   ~3.8 GB    仅 BGE-M3 + 系统
```

**调度策略**:
| 场景 | BGE-M3 | Reranker | PaddleOCR | 显存 |
|------|:---:|:---:|:---:|------|
| 空闲 | ✅ 常驻 | — | — | 3.8 GB |
| 用户提问 | ✅ | ✅ 临时加载 | — | 5.8 GB → 用完释放 |
| 入库纯文字 | ✅ | — | — | 3.8 GB |
| 入库含扫描页 | ✅ | — | ✅ 2进程 | 8 GB → 子进程退出释放 |
| 提问+入库同时 | ✅ | ✅ | ✅ 1进程 | 7-8 GB ✓ |

### 2.8 OCR 引擎 (`ingestion/ocr_engine.py`)

**类**: `OCREngine` — PaddleOCR 子进程隔离封装

通过子进程运行 PaddleOCR，解决 paddlepaddle GPU 版与环境冲突。OCR 子进程使用 PPOCRLabel venv 的 Python 解释器（paddlepaddle-gpu 3.2.2 + paddleocr 3.6.0 兼容配对）。

**架构**:
```
主进程 (conda base, protobuf 5.x + pymilvus)
    │
    │  _get_ocr_python() → PPOCRLabel .venv python
    │  _get_ocr_env()    → PYTHONPATH="" + PROTOCOL_BUFFERS=python
    │  subprocess.run()
    ▼
OCR 子进程
    │  sys.stdout → stderr (模型加载日志)
    │  _orig_stdout_fd → 干净 JSON 通道
    │
    ├─ import torch (先加载，固定 DLL)
    ├─ import paddle → paddleocr.PaddleOCR
    └─ PaddleOCR(lang="ch", use_textline_orientation=True, device="gpu")
```

**stdout/stderr 分离机制**:
- CLI 入口 `__main__` 将 `sys.stdout` 重定向到 `sys.stderr`
- 保存 `_orig_stdout_fd = os.fdopen(os.dup(1))` 作为原始 stdout
- PaddleOCR 初始化日志（ANSI 彩色输出、模型加载信息）→ stderr
- `_stdout_json()` 通过 `_orig_stdout_fd` 输出 JSON → 主进程 `capture_output` 拿到干净数据
- `_parse_json_stdout()` 兼容处理: 从多行混合输出中提取 JSON（从后往前找 `{` 行）

**PaddleOCR 2.x/3.x 兼容**:
```python
# 3.x: OCRResult dict
if isinstance(r0, dict) and "rec_texts" in r0:
    texts = r0["rec_texts"]
    scores = r0["rec_scores"]

# 2.x: [[box, (text, conf)], ...]
elif isinstance(r0, list):
    for box, (text, confidence) in r0:
        ...
```

**批量子进程模式** (`--batch`):
- 主进程将图片路径列表通过 stdin JSON 传给子进程
- 子进程一次性加载模型，批量处理所有图片
- DPI 150: GPU ~5-7s/页 (PP-OCRv5_server)

**大文件分批 + 动态超时**:
- 每批 ≤50 页，避免单次子进程超时和显存溢出
- 超时 = `max(300, chunk_pages × 15s)`，按页数自动伸缩

**API**:
```python
engine = OCREngine(lang='ch', dpi=150, use_subprocess=True, use_gpu=True)

# 单图 OCR
result = engine.ocr_page("/path/to/page.png")

# PDF 批量 OCR (内部自动分批)
result = engine.ocr_pdf_pages("/path/to/doc.pdf", pages=[0..137])
```

### 2.9 文件处理器 (`ingestion/file_processor.py`)

**类**: `FileProcessor` — 完整入库管道 + 双槽并行调度

```
process(file_path, domain, category)
    │
    ├─ compute_hash() → 查注册表 → 已入库则跳过
    │
    ├─ [PDF + OCR 启用] _process_pdf_progressive()
    │   │
    │   ├─ PDFParser.parse() → 检测 needs_ocr_pages
    │   ├─ ThreadPoolExecutor(2) → 2个OCR子进程并行
    │   │   Worker1: ocr_pages(batch1) ─┐
    │   │   Worker2: ocr_pages(batch2) ─┤ 同时跑!
    │   │                                │
    │   ├─ queue.Queue ← 分块完成入队 ←┘
    │   └─ Thread: 嵌入+insert (与OCR并行消费队列)
    │
    └─ [通用路径] _parse_file() → _embed_and_insert()
        │
        ├─ Step 1: 解析
        │   ├─ PDF: PDFParser → Chunker
        │   ├─ DOC: Word COM (win32com) → LibreOffice → olefile → antiword
        │   ├─ DOCX: python-docx (段落+表格) → Chunker
        │   ├─ WPS: 6级回退: docx→zip→LibreOffice→olefile→WPS COM→Word COM
        │   ├─ XLSX: openpyxl → Chunker
        │   └─ TXT/MD: 直接读取 → Chunker
        │
        ├─ Step 2: 嵌入
        │   Embedder.encode(embedding_texts) — GPU CUDA
        │
        └─ Step 3: 入库
            MilvusStore.insert(chunks, dense_vecs, sparse_vecs)
```

**双槽并行效果**:
```
138页全扫描 (10批, 每批15页):
  串行: 10×75s + 嵌入56s = 806s
  双槽: 5×75s + 嵌入56s(与OCR并行) ≈ 430s  快 47%
```

**docx 表格提取**:
```python
for table in doc.tables:
    for row in table.rows:
        cells = [p.text for p in cell.paragraphs if p.text.strip()]
        rows_text.append(" | ".join(cells))
    # → "[表格1]\n列A | 列B | 列C"
```

**三端一致性**:
- `list_files(check_existence=True)`: 返回 `file_exists` 字段
- `delete(remove_file=True)`: 同步清理 uploads 下物理文件
- `sync_orphans()`: 扫描物理文件已消失的记录 → 清理向量+标记 deleted
- 异常中断恢复: `delete_by_file_hash(file_hash)` 清理已插入 chunk

### 2.10 LLM 推理引擎 (`generation/llm_engine.py`)

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

### 2.11 对话管理 (`generation/conversation_manager.py`)

**类**: `ConversationManager`

- 内存存储 + 日后可迁移 SQLite
- 自动压缩: 超过 `max_context_tokens * 0.85` 时，旧消息合并为摘要
- 保留最近 `keep_detail_rounds=3` 轮完整内容
- 所有时间戳使用北京时间 `Asia/Shanghai`

### 2.12 Gradio UI (`ui/app.py`)

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

### GET /gpu

GPU 显存状态（`新增`）:

```json
{
  "total_mb": 12282,
  "used_mb": 3833,
  "free_mb": 8448,
  "usage_pct": 31.2,
  "ollama_busy": false
}
```

### POST /files/sync

一致性校验（`新增`）:

```
POST /files/sync?dry_run=false  → 执行清理
POST /files/sync?dry_run=true   → 仅扫描
```

### DELETE /files/{identifier}

```
DELETE /files/{identifier}?remove_file=true   → 同时清理物理文件
DELETE /files/{identifier}?remove_file=false  → 仅删向量+注册表
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

`config.yaml` 位于项目根目录，所有路径相对解析。关键配置节：

```yaml
ocr:
  use_gpu: true              # GPU 推理 (~5s/页 vs CPU ~15s/页)
  dpi: 150
  gpu_memory:
    min_free_vram_mb: 2500   # 入库所需最小空闲显存
    poll_interval_s: 5       # 显存轮询间隔
    max_wait_s: 600          # 最大等待时间

embedding:
  provider: "sentence_transformers"
  model_name: "BAAI/bge-m3"
  device: "cuda"

reranker:
  provider: "flagembedding"
  model_name: "BAAI/bge-reranker-v2-m3"
  device: "cuda"
```

### 5.2 模型运行位置

| 模型 | 运行位置 | 加载策略 | 显存 |
|------|----------|----------|------|
| BAAI/bge-m3 | GPU (sentence_transformers) | 常驻 | ~2.0 GB |
| BAAI/bge-reranker-v2-m3 | GPU (FlagEmbedding) | 按需加载/卸载 | ~2.0 GB |
| PaddleOCR PP-OCRv5_server | GPU (PPOCRLabel venv) | 入库时子进程加载，退出释放 | ~2.5 GB |
| qwen3.5:4b | GPU (Ollama) | 独立进程 | ~4.0 GB |

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
| 单 chunk 嵌入 | ~200ms (GPU CUDA, batch=32) |
| 检索耗时 | ~3-4s (QueryAnalyze+HybridSearch+Reranker加载+Rerank) |
| LLM token/s | ~112 t/s (qwen3.5:4b, think=false) |
| 问答总耗时 | ~8-15s (搜索4s + LLM生成5-12s) |
| OCR 扫描页 | ~5-7s/页 (GPU, PP-OCRv5_server) |
| 138页全扫描件 | ~7-8 分钟 (双槽并行) |
| 粗召回候选数 | 50 |
| 精排结果数 | 15 (可配置) |
| GPU | NVIDIA RTX 4070 SUPER (12GB) |
| 空闲显存 | ~8.4 GB (常态仅 BGE-M3 + 系统) |
