# RAG 知识库系统

基于 RAG (Retrieval-Augmented Generation) 架构的企业文档知识库问答系统。支持大规模文档入库、混合检索、重排序、基于大模型的智能问答，前端提供知识库管理、智能问答、文档预览等功能。

## Excel 工作簿分析 (集成 Excel Workbook Service)

顶部导航「**Excel 分析**」接入独立微服务 `excel-workbook-service`(抽取自 DB-GPT 0.8.1),提供:

**上传 Excel → 审核草稿(改表名/字段/类型/单元格/确认警告)→ 确认建 SQLite 库 → 自然语言 SQL 问答 → 生成 HTML 分析报告**

- 后端 `src/api/main.py` 末尾挂载了 `src/api/excel_proxy.py`(`/excel/*` 反向代理)。
- 微服务默认 `http://127.0.0.1:8090`,可用环境变量 `EXCEL_SERVICE_BASE` 覆盖;Excel 专用 LLM 的 API Key 用 `EXCEL_LLM_API_KEY`(见 `.env.example`)。
- 微服务独立安装/启动,参见其 `README.md`:

```bash
cd D:\git\excel-workbook-service
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python run.py            # 127.0.0.1:8090
# 编辑 config.yaml 的 excel.llm.base_url 指向可达的 OpenAI 兼容 /v1 服务
```

- 前端组件在 `src/ui-vue2/src/excel/`(工作台/上传/审核/问答/报告 iframe),`api.js` 与 `sse.js` 已扩展类型化 SSE(`sql/table/html/done`)。
- 原有 `/upload`、`/ask`、`/conversations` 等路由完全不受影响。

## 架构

```
┌─────────────┐     ┌──────────────────────────────────────────────┐
│  Vue3 前端  │────▶│                 FastAPI 后端                   │
│  (ui-vue2)  │     │  /upload /search /ask /files ...              │
└─────────────┘     └──────┬──────────────┬──────────────┬─────────┘
                           │              │              │
                   ┌───────▼──────┐ ┌─────▼────────┐ ┌───▼─────────┐
                   │ 文档解析      │ │ 混合检索      │ │  LLM 生成   │
                   │ PyMuPDF       │ │ Qwen dense   │ │ OpenAI 兼容 │
                   │ + PaddleOCR-VL│ │ BGE-M3 sparse│ │ 服务        │
                   └──────────────┘ │ BM25 + rerank│ └─────────────┘
                                    └───────────────┘
```

- **文档解析**：PDF、CEB 和 PNG/JPEG 等页面通过外部 **PaddleOCR-VL** 服务结构化识别（阅读顺序 + Markdown 表格 + 标题层级 + bbox），CEB 使用 Apabi 原生渲染为多页 PNG，不经过 PDF。
- **混合检索**：稠密向量继续使用 ModelScope 的 `Qwen3-Embedding-0.6B`；稀疏向量由本地 CPU `BGE-M3` 生成并写入 Milvus；另有带语料 IDF 和文档长度归一化的 BM25 关键词分支，最终经过本地 CPU `BGE-Reranker-v2-m3` 重排。
- **LLM 生成**：调用任意 **OpenAI 兼容** 的 `/v1/chat/completions` 服务（vLLM / Xinference / LM Studio / PaddleX serving 等），无需本地部署模型。
- **前端**：Vue 3 + Element Plus（`src/ui-vue2`，默认 5174 端口）。

文档预览同时支持检索文本和版面还原：PDF 按页缓存 OCR 版面块，预览时生成独立页面；标题目录支持章节层级导航，公式、表格、公式编号和目次/Contents 使用统一渲染规则。详细链路见 [`docs/PDF完整解析入库预览链路.md`](docs/PDF完整解析入库预览链路.md)、[`docs/PNG上传解析入库后处理预览流程.md`](docs/PNG上传解析入库后处理预览流程.md) 和 [`docs/开发日志.md`](docs/开发日志.md)。

## 快速开始

### 环境要求

- Python 3.10+
- 一个 OpenAI 兼容的 LLM 服务（可选本地 vLLM，也可用云服务）
- 一个 OpenAI 兼容的 PaddleOCR-VL 服务（扫描件识别，可选，无扫描件可跳过）
- NVIDIA GPU（可选；Qwen 稠密向量走 API，BGE-M3 稀疏向量和重排模型可用 CPU，GPU 仅用于加速）

### 1. 安装后端依赖

```bash
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/macOS

pip install -r requirements.txt

# 安装 PyTorch (按 CUDA/CPU 选其一):
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.x
# pip install torch   # CPU
```

### 2. 配置

编辑 `config.yaml`：

```yaml
llm:
  provider: "openai"
  openai:
    base_url: "http://<LLM服务地址>/v1"   # 如 http://192.168.0.201:18000/v1
    model: "/data/models/Qwen/Qwen3-4B"
    api_key: ""                            # 本地服务留空; 云服务填真实 key

ocr:
  provider: "vl"
  vl:
    base_url: "http://<OCR-Pipeline服务地址>" # 如 http://192.168.0.201:8001
    protocol: "pipeline"
    endpoint: "/ocr"
  compare:
    legacy_base_url: "http://<兼容 OCR 服务地址>" # 可选，如 192.168.0.201:8080

ceb:
  enabled: true
  renderer_script: "scripts/ceb_render_pages.ps1"
  apabi_dir: "D:/Apabi reader"
  render_width: 800
  render_height: 1000

embedding:
  provider: "openai"                         # Qwen dense API
  dimensions: 1024
  sparse_provider: "bge_m3"                  # 本地 BGE-M3 lexical sparse
  sparse_device: "cpu"
  openai:
    base_url: "https://api-inference.modelscope.cn/v1"
    model: "qwen/Qwen3-Embedding-0.6B"
  hf_home: "D:/git/RongNengRAG/data/hf_cache"

reranker:
  provider: "flagembedding"
  model_name: "BAAI/bge-reranker-v2-m3"
  device: "cpu"

retrieval:
  bm25_enabled: true
  dense_weight: 0.7
  sparse_weight: 0.3
  bm25_weight: 0.25
```

首次运行会自动下载本地 BGE-M3 稀疏模型和 BGE-Reranker-v2-m3 重排模型到 `embedding.hf_home`；当前默认使用 CPU。Qwen 稠密向量仍调用 `embedding.openai` 配置的远程兼容 API。

### 3. 启动后端

```bash
cd src
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API 文档：http://localhost:8000/docs

### 4. 启动前端

```bash
cd src/ui-vue2
npm install
npm run dev -- --host 0.0.0.0  # 局域网访问：http://<服务器局域网IP>:5174
```

若后端不在本机，指定后端地址：

```bash
npm run dev -- --host 0.0.0.0     # 或用环境变量
VITE_API_BASE=http://192.168.x.x:8000 npm run dev -- --host 0.0.0.0
```

### 5. 入库文档

```bash
# 单文件
python scripts/build_index.py add-file --file "D:/docs/sample.pdf"

# 批量目录 (递归扫描)
python scripts/build_index.py add-dir --dir "D:/docs"

# 查看统计
python scripts/build_index.py summary

# 旧库迁移：停掉后端后，将 hashed-TF sparse 改为 BGE-M3，保留原 dense 向量
python scripts/rebuild_sparse_vectors.py
```

或通过前端页面上传。

## 主要 API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/upload` | POST | 单文件上传入库 |
| `/upload/batch` | POST | 批量上传 |
| `/files` | GET | 列出已入库文件 |
| `/files/{id}/content` | GET | 获取文档全文 + 分块 (含页码) |
| `/files/{id}/ocr-compare` | GET | 逐页对比 8001 Pipeline 与 8080 兼容 OCR (诊断，不写入库) |
| `/files/{id}/download` | GET | 按原始文件名和格式下载原文件 |
| `/search` | POST | 纯检索 (不生成回答) |
| `/ask` | POST | RAG 完整问答 |
| `/ask/stream` | POST | SSE 流式问答 |
| `/stats` | GET | 知识库统计 |

## 目录结构

```
config.yaml                # 全局配置 (LLM / OCR / 分块 / 检索)
src/
  api/main.py              # FastAPI 后端
  ingestion/               # 文档解析 + 分块 + 嵌入
    pdf_parser.py          # PDF 页面解析与 OCR 调度
    ceb_renderer.py        # Apabi CEB 原生分页 PNG 渲染适配
    vl_ocr.py              # PaddleOCR-VL 协议适配、版面渲染、目录提取
    chunker.py             # 分块引擎 (按页分块, 保留页码)
    embedder.py            # Qwen 稠密 API + 本地 BGE-M3 稀疏向量
    milvus_store.py        # Milvus Lite 存储
  retrieval/               # BM25/向量混合检索 + 重排序
    bm25_index.py          # 带 IDF/文档长度归一化的 BM25
    reranker.py            # BGE-Reranker-v2-m3 本地 CPU/GPU 重排
  generation/              # LLM 生成
    providers/openai_compat_provider.py   # OpenAI 兼容 LLM
  ui-vue2/                 # Vue 3 前端 (5174)
scripts/build_index.py     # 命令行入库工具
scripts/rebuild_sparse_vectors.py # 旧 sparse 向量迁移为 BGE-M3
docs/PNG上传解析入库后处理预览流程.md  # 图片/PDF 上传、解析、入库与预览链路
docs/PDF完整解析入库预览链路.md          # 已验证的 PDF 解析、入库与版面预览完整链路
docs/开发日志.md            # 最近一轮 OCR、PDF 和预览改造记录
```

## License

本项目为开源项目，License 待定。
