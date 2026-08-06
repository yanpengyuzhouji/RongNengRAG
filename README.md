# RAG 知识库系统

基于 RAG (Retrieval-Augmented Generation) 架构的企业文档知识库问答系统。支持大规模文档入库、混合检索、重排序、基于大模型的智能问答，前端提供知识库管理、智能问答、文档预览等功能。

## 架构

```
┌─────────────┐     ┌────────────────────────────────────────┐
│  Vue3 前端  │────▶│             FastAPI 后端                 │
│  (ui-vue2)  │     │  /upload /search /ask /files ...        │
└─────────────┘     └──────┬──────────┬───────────┬───────────┘
                           │          │           │
                   ┌───────▼──┐  ┌────▼────┐  ┌───▼─────────┐
                   │ 文档解析  │  │ 向量检索  │  │  LLM 生成    │
                   │ PyMuPDF   │  │ Milvus   │  │ OpenAI 兼容  │
                   │ + PaddleOCR-VL │  │ 混合检索  │  │ 服务        │
                   └──────────┘  └─────────┘  └─────────────┘
```

- **文档解析**：PyMuPDF 提取文字 PDF；扫描件通过外部 **PaddleOCR-VL** 服务（OpenAI 兼容协议）结构化识别（阅读顺序 + markdown 表格 + 标题层级），按页入库。
- **向量检索**：BGE-M3 生成稠密 + 稀疏向量（本地 GPU），Milvus Lite 混合检索（BM25 稀疏 + 稠密），BGE-Reranker 交叉编码器精排。
- **LLM 生成**：调用任意 **OpenAI 兼容** 的 `/v1/chat/completions` 服务（vLLM / Xinference / LM Studio / PaddleX serving 等），无需本地部署模型。
- **前端**：Vue 3 + Element Plus（`src/ui-vue2`，默认 5174 端口）。

## 快速开始

### 环境要求

- Python 3.10+
- 一个 OpenAI 兼容的 LLM 服务（可选本地 vLLM，也可用云服务）
- 一个 OpenAI 兼容的 PaddleOCR-VL 服务（扫描件识别，可选，无扫描件可跳过）
- NVIDIA GPU（可选，嵌入模型可退 CPU；推荐 GPU 提速）

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
    model: "qwen2.5-7b-instruct"
    api_key: ""                            # 本地服务留空; 云服务填真实 key

ocr:
  provider: "vl"
  vl:
    base_url: "http://<OCR-VL服务地址>"     # 如 http://192.168.0.201:8080
    model: "paddleocr-vl-1.6"
```

首次运行会自动下载 BGE-M3 / BGE-Reranker 模型（国内可用 HF 镜像，见 `config.yaml` 的 `hf_endpoint`）。

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
npm run dev        # 默认 http://localhost:5174
```

若后端不在本机，指定后端地址：

```bash
npm run dev -- --host     # 或用环境变量
VITE_API_BASE=http://192.168.x.x:8000 npm run dev
```

### 5. 入库文档

```bash
# 单文件
python scripts/build_index.py add-file --file "D:/docs/sample.pdf"

# 批量目录 (递归扫描)
python scripts/build_index.py add-dir --dir "D:/docs"

# 查看统计
python scripts/build_index.py summary
```

或通过前端页面上传。

## 主要 API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/upload` | POST | 单文件上传入库 |
| `/upload/batch` | POST | 批量上传 |
| `/files` | GET | 列出已入库文件 |
| `/files/{id}/content` | GET | 获取文档全文 + 分块 (含页码) |
| `/files/{id}/download` | GET | 下载原文件 |
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
    pdf_parser.py          # PDF 解析 (fitz)
    vl_ocr.py              # PaddleOCR-VL 结构化识别客户端
    chunker.py             # 分块引擎 (按页分块, 保留页码)
    embedder.py            # BGE-M3 嵌入
    milvus_store.py        # Milvus Lite 存储
  retrieval/               # 混合检索 + 重排序
  generation/              # LLM 生成
    providers/openai_compat_provider.py   # OpenAI 兼容 LLM
  ui-vue2/                 # Vue 3 前端 (5174)
scripts/build_index.py     # 命令行入库工具
```

## License

本项目为开源项目，License 待定。
