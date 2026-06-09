# 开发日志

## 2026-06-09

### PaddleOCR 扫描件 PDF 识别

- **新增** `src/ingestion/ocr_engine.py` — PaddleOCR 引擎（子进程隔离模式）:
  - 通过子进程运行 OCR，解决 paddlepaddle protobuf 3.x 与 pymilvus protobuf 5.x 版本冲突
  - 主进程: `protobuf>=5.27.2` (pymilvus 正常) / OCR 子进程: `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`
  - 支持 `--batch` 批量模式: 一次子进程加载模型，处理多张图片（4x 加速）
  - 单页模式: `ocr_page(image_path)` → 子进程单图 OCR
  - PDF 批量模式: `ocr_pdf_pages(pdf_path, pages)` → 渲染 PNG → 子进程批量识别
  - `OCREngine` 类: `use_subprocess=True` 自动隔离模式
  - 首次调用 ~10s (模型加载)，后续 ~1.5s/页 (CPU)
  - 子进程超时保护 600s / 120s
- **修改** `src/ingestion/pdf_parser.py`:
  - `__init__` 新增 `ocr_config` 参数
  - 新增 `_get_ocr_engine()` — 延迟加载 OCR 引擎
  - 新增 `ocr_pages(filepath, pages)` — 批量 OCR PDF 页面
  - 新增 `ocr_page(image_path)` — 单图 OCR
- **修改** `src/ingestion/file_processor.py`:
  - `__init__` — 传递 OCR 配置给 PDFParser
  - `_parse_file` PDF 分支 — 解析后检测 `needs_ocr_pages`，自动调用 OCR 并回填文本
  - OCR 文本与原 fitz 文本智能合并（有则追加）
- **修改** `src/api/main.py`:
  - 服务启动时异步预热 OCR 子进程（首次调用不等待）
- **修改** `config.yaml` — 新增 `ocr` 配置节:
  ```yaml
  ocr:
    enabled: true          # 启用 OCR
    provider: "paddleocr"  # PaddleOCR
    lang: "ch"             # 中文
    dpi: 200               # PDF 渲染 DPI
    min_text_chars: 50     # 触发 OCR 阈值
    use_subprocess: true   # 子进程隔离
  ```
- **效果**:
  - 208 页 PDF: 4 页标记为扫描件 → 批量 OCR 6s (1.5s/页)
  - 封面页识别: "国家电网有限公司 / 十八项电网重大反事故措施 / （修订版）" (99.8% 置信度)
  - 空白页正确跳过，不消耗额外 OCR 时间
  - 不影响原有文字型 PDF 的解析速度（仅在 `needs_ocr` 时触发）

### .wps 文件解析支持

- **修改** `src/ingestion/file_processor.py`:
  - `_parse_file` 新增 `.wps` 分支
  - 新增 `_parse_wps_file()` — 6 级回退解析管道
  - 新增 `_parse_wps_via_docx()` — 新版 WPS (ZIP+XML) 用 python-docx 读取
  - 新增 `_parse_wps_via_zip()` — 备选: 直接解压 ZIP 提取 XML 文本
  - 新增 `_parse_wps_via_wps_com()` — WPS Office COM 自动化
  - 后 3 级复用 .doc 管道 (LibreOffice / olefile / MS Word COM)
- **修改** `src/api/main.py` — 上传白名单添加 `.wps`

---

## 2026-06-08

### 文件注册表识别 + 完整文档注入

- **新增** `src/retrieval/file_registry.py` — `FileRegistry` 类:
  - 从 SQLite `file_registry` 表读取已入库文件元数据，60 秒缓存
  - `detect_files_in_query(query)` — 四阶段多策略文件名检测:
    - **策略1**: 正则提取 query 中的文件名候选（书名号、引号、扩展名模式、文档编号）
    - **策略2**: 双向匹配 — 文件名 in query + query 关键词 in 文件名（解决短 query 匹配长文件名问题）
    - **策略3**: 文档编号精确匹配（闽电发展〔2015〕241号、GB 50060-2008 等）
    - **策略4**: "会议材料之X" / "会议材料X" 系列匹配 — 含中文数字序数归一化（一→1）
  - 支持五种自然语言写法: `01会议材料之一` / `会议材料之一` / `01会议材料一` / `会议材料一` / `材料一`
  - 中文→阿拉伯数字映射: 一～二十 → 1~20，含 `zfill(2)` 对齐 "01" 格式
  - `_extract_query_phrases(query)` — 将 query 拆解为关键词片段并生成数字变体
  - 返回 `FileMatchResult`（含 `match_type`、`match_score`、`match_text`），去重 + 按分数降序

- **修改** `src/ingestion/milvus_store.py`:
  - 新增 `query_by_file_path(file_path)` — 按 `file_path` 精确查询文件的完整 chunks
  - 新增 `query_by_file_hash(file_hash)` — 按 `chunk_id LIKE "{hash}%"` 前缀查询（主方案，最可靠）
  - 返回结果按 `page_num` + `chunk_index` 排序

- **修改** `src/retrieval/retriever.py`:
  - `__init__` 新增 `self.file_registry` 实例
  - 新增 `detect_file_in_query(query)` — 委托 FileRegistry 返回最佳匹配
  - 新增 `get_full_document(file_path, file_hash)` — 从 Milvus 拉取完整文档、格式化输出（含页眉+页码导航）
  - 新增 `build_context_with_file_injection(query, search_results, max_chunks)` — LLM 上下文构建核心:
    - 检测到文件名: 完整文档前置 + 强化聚焦指令 + 补充检索（排除同系列 + 最多 2 个）
    - 未检测到: 正常检索格式
  - 新增 `_extract_series_key(filename)` — 从文件名提取"系列标识"
    - "01会议材料之一..." → `会议材料之`
    - 用于排除同系列文件干扰（查询材料01时排除02-07）
  - 聚焦指令强化: `"请只基于上述完整文档内容回答，不要引用其他文件"`
  - 补充检索结果标注: `"以下内容仅供背景了解，回答时请勿引用"`
  - 补充最多保留 2 个文件（原 5+ 个）
  - 新增 debug 日志输出匹配过程

- **修改** `src/api/main.py`:
  - `/ask` 和 `/ask/stream` 端点改用 `r.build_context_with_file_injection()` 替代旧版 `_build_focused_context`
  - 边界处理: 检索无结果但 query 含文件名时，从注册表拉取完整文档再送 LLM
  - 保留 `_build_focused_context_fallback` 作为纯文本回退方案

- **效果**:
  - 查询 "会议材料一总结" → 精确匹配 01 文件，排除 02-07，回答仅含材料一内容
  - 查询含文件名 → score=1.0 精确匹配，注入完整文档（22k+ chars）
  - 查询无文件名 → 不触发，正常检索流程
  - 同系列文件 100% 排除，不再混淆

### Qwen3 强制思考链优化 — 终极方案

- **根因**: Qwen3 (4b/8b/14b) 和 DeepSeek-R1 是推理模型，强制生成 thinking tokens（占 85-95% 输出），`enable_thinking=false` 和 `disable_cot=true` 均对 Qwen3 无效
- **方案演进**:
  1. Ollama OpenAI兼容API `/v1/chat/completions` → reasoning 字段被 openai 库丢弃 → 改用原生 `/api/chat`
  2. `disable_cot=true` 顶层参数 → Qwen3 不接受，反而更慢
  3. `ollama create qwen3:4b-nothink` 自定义模板移除 think 标签 → 有效但只是隐藏，仍在生成
  4. **最终方案**: 升级到 Ollama 最新版 + `qwen3.5:4b`，原生支持 `think: false` → 完全关闭思维链
- **修改**: `ollama_provider.py` — 新增 `think` 参数，`payload["think"]=false` 关闭思考；流式直接输出无需缓冲
- **预期效果**: 32k 上下文下 25-35 t/s，单次 RAG 问答 15-25 秒（vs qwen3:4b 的 70-90 秒）

### LLM Provider 模式重构 + 百炼 API 支持

- **新增** `src/generation/providers/` 包:
  - `base.py` — `BaseProvider` 抽象基类 (`generate` + `generate_stream`)
  - `ollama_provider.py` — 迁移原有 Ollama 逻辑到独立 Provider，改用原生 `/api/chat` 端点
  - `bailian_provider.py` — 阿里云百炼 DashScope，OpenAI 兼容接口
- **修改** `llm_engine.py` — 重构为 Provider 模式 facade，支持 `generate_chat()` 多轮对话 + `generate_chat_stream()` 流式
- **修改** `config.py` — `load_config()` 自动加载 `.env` 环境变量
- **新增** `.env.example` — API Key 模板

### 多轮对话 + 上下文记忆系统

- **新增** `src/generation/conversation_manager.py`:
  - 会话管理: create/list/get/delete，每会话独立历史
  - 压缩策略: 超窗口时保留最近 3 轮详细信息，旧消息压缩为摘要
  - 所有时间戳使用北京时间 (Asia/Shanghai)
- **修改** `api/main.py`:
  - `AskRequest` 添加 `conversation_id` 字段
  - 新增 `POST/GET/DELETE /conversations` CRUD 端点
  - `/ask` 支持多轮上下文加载
- **修改** `prompt_templates.py` — 新增 `get_system_prompt()` 系统提示词

### SSE 流式生成 + 会话侧边栏 + 会话隔离

- **修改** `api/main.py` — 新增 `POST /ask/stream` SSE 端点:
  - 检索完成后立即发送 `searching` → `thinking` 心跳
  - 模型产出 token 实时推送到前端
- **修改** `ui/app.py` — 大幅重构:
  - **新布局**: 左侧会话侧边栏 + 主区域 Chatbot
  - 会话管理: 新建/切换/删除，`gr.State` 维护会话隔离
  - 流式展示: 接收 SSE 事件实时更新 Chatbot
  - 北京时间显示: 每条消息带时间戳
  - 保留原有文件管理/检索/统计功能
- **新增** API 流式心跳机制: thinking 阶段显示 "💭 思考中..." 动画

### 检索置信度计算

- **修改** `reranker.py` — 为每个结果附加 `_rerank_score` 并映射为 `confidence` (0-1)
- **修改** `retriever.py` — `RetrievalResult` 新增 `confidence: float` 字段
- **修改** `api/main.py` — `SearchResultItem` 新增 `confidence` 字段
- **修改** `ui/app.py` — 搜索结果显示置信度百分比 + 可视化进度条
- **修改** `config.yaml` — 新增 `retrieval.confidence` 配置

### .doc 文件解析

- **修改** `file_processor.py`:
  - 新增 `_parse_doc_file()` 方法，按优先级: Word COM → LibreOffice → olefile → antiword → docx2txt
  - 新增 `_parse_doc_via_win32()` — 通过 `win32com` 自动化本机 Word 提取文本
  - `_parse_file()` 添加 `.doc` 分支
- **效果**: 测试文件 "附件1福建省变配电工程消防设计技术要点.doc" → 37 chunks, 79,664 chars

### 文件详情查看修复

- **修改** `api/main.py` — 新增 `GET /files/{identifier}` 端点
- **修改** `ui/app.py` — `view_file_details()` 直接调 API 查找，不再依赖过期缓存

### 文档列表类目换行修复

- **修改** `ui/app.py` `_build_file_tree()` — 每个类目标题前插入空行，避免 markdown 渲染连在一起

### 文件管理标签页加载修复

- **修改** `ui/app.py` — 文件管理/检索/统计 Tab 补齐 `demo.load` 事件，页面打开自动加载

---

## 2026-06-05

### 路径配置项目化

- **新增** `src/config.py` — 统一配置加载模块
  - 自动检测项目根目录（向上查找 `config.yaml`）
  - 所有相对路径自动解析为绝对路径
  - `ensure_data_dirs()` 自动创建数据目录
  - `get_config_path()` 返回默认配置文件路径
- **修改** `config.yaml` — 所有路径改为相对路径
  - `knowledge_base: ""` 暂时留空，批量导入时再配置
  - `metadata_db: "data/file_metadata.db"`
  - `milvus_db: "data/milvus_lite.db"`
  - `uploads_dir: "data/uploads"`
  - `parsed_cache: "data/parsed_cache"`
  - `models_dir: "data/models"`
- **更新 8 个模块** — 全部改用 `from config import load_config`
  - `milvus_store.py`, `embedder.py`, `file_processor.py`, `chunker.py`
  - `file_walker.py`, `query_analyzer.py`, `reranker.py`, `retriever.py`
  - `llm_engine.py` (llama_cpp 路径也改为项目相对)
  - `api/main.py` (添加 startup 事件初始化数据目录)
- **数据存储位置**: `D:/rag-system/` → 项目内 `data/`
- **部署简化**: 克隆项目后无需创建外部目录或复制配置文件

### 文档与项目管理

- **新增** `README.md` — 项目概述、技术栈、快速开始、项目结构、架构图
- **新增** `TECHNICAL.md` — 详细技术文档（模块详解、API 接口、数据流、性能参数）
- **新增** `CHANGELOG.md` — 本开发日志

---

## 2026-06-04

### 启动项目与依赖修复

- **安装缺失依赖**: `pymilvus`, `milvus-lite`, `FlagEmbedding`, `PyMuPDF`, `python-pptx`, `xlrd`, `rarfile`, `faiss-cpu`
- **创建数据目录**: `D:/rag-system/data/` (uploads, parsed_cache, milvus_lite.db)
- **配置 LLM 模型**: 将 `qwen2.5:7b-instruct-q4_K_M` 改为已安装的 `qwen3:8b`
- **复制配置文件**: `config.yaml` → `D:/rag-system/config.yaml`

### HuggingFace 离线迁移 — 全部改为本地 Ollama

#### 嵌入引擎 (`ingestion/embedder.py`)
- 默认使用 Ollama `bge-m3:latest` 的 `/api/embed` 批量 API
- 保留 `sentence_transformers` 作为回退方案
- 添加 `_init_ollama()` 连接验证
- 修复 `from chunker import Chunk` 无效导入问题

#### 重排序器 (`retrieval/reranker.py`)
- 默认使用 Ollama 嵌入 + 余弦相似度 + 元数据加权
- 批量嵌入 Query + 候选文档计算相似度
- 保留 FlagEmbedding 交叉编码器作为回退
- 添加 `rerank_without_model` 纯元数据排序兜底
- 修复所有 emoji → ASCII 以兼容 Windows GBK 终端

#### 配置文件 (`config.yaml`)
- `embedding.provider`: `"ollama"` (默认)
- `reranker.provider`: `"ollama"` (默认)
- `device`: `"cpu"` → `"cuda"` (GPU 加速)

### Milvus Lite 多进程冲突修复

#### 架构变更: Gradio UI → HTTP 客户端模式
- **根因**: Gradio 和 FastAPI 两个独立 Python 进程同时打开 Milvus Lite，LMDB 排他锁冲突
- **修复**: `ui/app.py` 完全重写为 FastAPI HTTP 客户端
  - 问答: `POST /ask`
  - 检索: `POST /search`
  - 上传: `POST /upload`
  - 管理: `GET /files`, `DELETE /files/{id}`
  - 统计: `GET /files/summary`

### 空集合检索报错修复

- `milvus_store.py:hybrid_search()`: 集合不存在时返回空列表而非抛异常
- `milvus_store.py:delete_by_file_hash()`: 集合不存在时跳过删除
- `milvus_store.py:insert()`: 添加 `_ensure_collection()` 自动创建集合
- `milvus_store.py:_connect()`: 连接失败时自动清理残留 `LOCK` 文件并重试

### Pydantic 校验错误修复
- `file_processor.py:_build_file_meta()`: 所有 `None` 值加 `or ""` / `or 0` 兜底
- `api/main.py:UploadResponse`: `domain`, `category`, `doc_number` 接收 `None` 导致 422 错误

### pymilvus 3.0 API 迁移
- `hybrid_search(rerank=...)` → `hybrid_search(ranker=...)`

### WinError 183 根本性修复
- **根因**: milvus-lite 3.0 在 Windows 上使用 `os.rename(tmp, target)` 写入 manifest.json，Windows 的 `os.rename` 不能覆盖已存在文件
- **修复**: `milvus_store.py` 顶部 monkey-patch `os.rename` → `_safe_rename`（内部用 `os.replace` 原子替换）
- **影响范围**: create_collection、create_index、insert、delete 全部不再受此 bug 影响

### 文件上传错误修复
- Gradio `upload_and_index()` 返回 2 值但 click 只定义 1 个输出 → 改为只返回结果字符串
- `upload_and_index()` 使用 `os.path.basename(f.name)` 提取原始文件名，避免乱码

### 检索结果数修复
- `retriever.py`: 构建响应时添加 `ranked[:top_k]` 硬限制
- 重排序失败时自动回退到 `rerank_without_model`

### 文件管理 UI 重写
- 新增文件夹树状视图：按域→类目→文件三级展示
- 新增文件选择器下拉框，支持按文件名搜索
- 新增"查看详情"功能：文件属性完整展示
- "刷新列表"默认排除已删除文件，显示文件数+chunk总数
- 页面加载时自动刷新文件列表
- 删除后自动清空下拉选择器并刷新树

### 搜索状态修复
- `milvus_store.py:insert()` 末尾添加 `load_collection()` 防止集合变为 `released` 状态
- `milvus_store.py:hybrid_search()` 开头添加 `load_collection()` 兜底
- `ui/app.py:_api()` 添加空响应检查和超时处理

### Windows 兼容性修复汇总
- 所有 emoji 打印替换为 ASCII 标签 (`[OK]`, `[WARN]`, `[info]`, `[drop]`, `[retry]`)
- `PYTHONIOENCODING=utf-8` 环境变量用于启动命令
- `os.rename` → `os.replace` 补丁
- LOCK 文件自动清理

---

## 2026-06-04 (Initial)

### 初始版本

- 完整的 RAG 系统脚手架
- 四域知识库结构 (变电/配电/送电输电/综合)
- 格式感知分块引擎 (语义/单页/全文)
- BGE-M3 嵌入 + 稀疏向量混合搜索
- 三阶段检索管道 (分析→召回→精排)
- 五种 LLM 提示词模板 (技术问答/文档查找/跨域对比/数值查规/通用)
- FastAPI REST API + Gradio UI
- 命令行索引工具 (`scripts/build_index.py`)
- SQLite 文件注册表 + Milvus Lite 向量库
