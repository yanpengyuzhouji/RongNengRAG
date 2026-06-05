# 开发日志

## 2026-06-05

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
