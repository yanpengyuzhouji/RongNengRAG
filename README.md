# 榕能电力审图知识库 RAG 系统

基于 RAG (Retrieval-Augmented Generation) 架构的电力设计审图智能问答系统，服务于变电、配电、送电输电三大专业域的 50,000+ 电力设计文件知识库。

## 技术栈

| 层次 | 技术 |
|------|------|
| 向量数据库 | Milvus Lite (内嵌式，无需 Docker) |
| 嵌入模型 | BAAI/bge-m3 (本地 sentence_transformers，稠密+稀疏) |
| 重排序 | BAAI/bge-reranker-v2-m3 (FlagEmbedding 交叉编码器) |
| LLM | Qwen3.5 4B (`qwen3.5:4b`，Ollama，think=false 关闭思考链) |
| 后端 API | FastAPI + Pydantic v2 + SSE 流式 |
| 前端 UI | Gradio 5.x (Chatbot 流式 + 会话侧边栏) |
| 文档解析 | PyMuPDF (PDF)、win32com (DOC)、python-docx (DOCX)、openpyxl (XLSX) |
| 多轮对话 | ConversationManager (上下文压缩 + 会话隔离) |
| API Key | .env 管理 (不入 git)，支持阿里云百炼 |

## 快速开始

### 环境要求

- Python 3.12+
- Ollama (用于本地模型推理)
- Windows / Linux / macOS

### 安装

```bash
# 1. 克隆项目
git clone <repo-url>
cd RongNengRAG

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装 Milvus Lite 内嵌引擎
pip install milvus-lite

# 4. 拉取 Ollama 模型 (Ollama 需更新到最新版)
ollama pull bge-m3:latest          # 嵌入模型 (Ollama备用)
ollama pull qwen3.5:4b             # 问答模型 (支持 think=false 关闭思考链)

# 5. 项目自带 config.yaml，首次启动自动创建 data/ 目录
#    无需手动创建路径
```

### 启动

```bash
# 终端 1: 启动 FastAPI 后端 (端口 8000)
cd src/api
PYTHONIOENCODING=utf-8 python main.py

# 终端 2: 启动 Gradio UI (端口 7860)
cd src/ui
PYTHONIOENCODING=utf-8 python app.py
```

打开浏览器访问 `http://localhost:7860`

> **注意**: 如遇 Windows GBK 编码问题，必须设置 `PYTHONIOENCODING=utf-8`。

### 导入文档

```bash
# 从目录批量导入
python scripts/build_index.py add-dir --dir "D:/知识库资料"

# 单个文件导入
python scripts/build_index.py add-file --file "D:/test.pdf"

# 查看入库统计
python scripts/build_index.py summary
```

## 项目结构

```
RongNengRAG/
├── config.yaml                  # 主配置文件
├── requirements.txt
├── .env.example                 # API Key 模板
├── README.md                    # 本文件
├── CHANGELOG.md                 # 开发日志
├── TECHNICAL.md                 # 技术文档
├── scripts/
│   └── build_index.py           # 命令行索引工具
└── src/
    ├── api/
    │   └── main.py              # FastAPI (REST + SSE + 多轮对话)
    ├── ui/
    │   └── app.py               # Gradio v3.0 (Chatbot + 会话侧边栏)
    ├── ingestion/               # 数据入库
    │   ├── file_processor.py    # 文件处理编排器 (.doc Word COM解析)
    │   ├── file_walker.py       # 知识库遍历 + 元数据提取
    │   ├── chunker.py           # 格式感知分块引擎
    │   ├── embedder.py          # BGE-M3 稠密+稀疏嵌入
    │   ├── milvus_store.py      # Milvus Lite 向量库
    │   └── pdf_parser.py        # PDF 文本提取
    ├── retrieval/               # 检索管道
    │   ├── retriever.py         # 三阶段检索 + 置信度
    │   ├── query_analyzer.py    # 查询分析器
    │   └── reranker.py          # 交叉编码器精排 + 元数据加权
    └── generation/              # 回答生成
        ├── llm_engine.py        # LLM Engine (Provider模式)
        ├── prompt_templates.py  # 领域提示词 + 系统提示
        ├── conversation_manager.py  # 多轮对话 + 上下文压缩
        └── providers/           # LLM Provider
            ├── base.py
            ├── ollama_provider.py   # Ollama 原生 /api/chat
            └── bailian_provider.py  # 阿里云百炼
```

## 架构概览

```
                     ┌──────────────┐
                     │  Gradio UI   │  (Chatbot + 会话侧边栏)
                     │  :7860       │
                     └──────┬───────┘
                            │ SSE Stream + HTTP
                     ┌──────▼───────┐
                     │   FastAPI    │  (后端服务)
                     │  :8000       │
                     └──┬───┬───┬──┘
          ┌─────────────┘   │   └──────────────┐
          ▼                 ▼                  ▼
   ┌──────────────┐  ┌──────────┐   ┌──────────────────┐
   │ FileProcessor│  │ Retriever│   │   LLMEngine      │
   │ (入库管道)    │  │ (3-stage)│   │ (Provider模式)    │
   └──────┬───────┘  └────┬─────┘   └────────┬─────────┘
          │               │                  │
          ▼               ▼                  ▼
   ┌──────────────┐  ┌──────────┐   ┌──────────────────┐
   │ Milvus Lite  │  │ Embedder │   │ OllamaProvider   │
   │ (向量库)     │  │ (BGE-M3) │   │ /api/chat        │
   └──────────────┘  └──────────┘   │ BailianProvider  │
                                    │ /v1/chat/...     │
                                    └──────────────────┘
```

## 检索管道

采用三阶段检索流程：

```
用户查询
    │
    ▼
阶段0: 查询分析 (QueryAnalyzer)
  ├─ 域分类 (变电/配电/送电输电/综合)
  ├─ 参数提取 (电压等级、设备类型、文档编号等)
  ├─ 查询类型判断 (技术问题/文档查找/跨域对比/数值查规)
  └─ 同义词扩展
    │
    ▼
阶段1: 粗召回 (Milvus Hybrid Search)
  ├─ Dense 向量搜索 (BGE-M3 稠密向量)
  ├─ Sparse 词法搜索 (BGE-M3 稀疏向量)
  └─ RRF 融合 (Reciprocal Rank Fusion)
    │ 50 candidates
    ▼
阶段2: 精排 (Reranker)
  ├─ 嵌入余弦相似度计算
  └─ 元数据加权 (文档编号/+1.5x, 标准规范/+1.2x, 域匹配/+1.1x)
    │ Top-K results
    ▼
LLM 生成回答 (带引用来源)
```

## 问答能力

| 查询类型 | 说明 | 示例 |
|----------|------|------|
| `domain_technical` | 专业领域技术问答 | "变电消防设计要求" |
| `document_lookup` | 按文档编号查找 | "闽电发展〔2015〕241号" |
| `cross_domain_comparison` | 跨域对比分析 | "变电和配电在接地要求上有什么区别" |
| `specification_lookup` | 数值/参数精确查规 | "10kV配电线路的安全距离是多少" |
| `general_qa` | 通用问答 | 其他 |

## 文档类型支持

| 格式 | 支持 | 说明 |
|------|------|------|
| PDF | ✅ | 文本提取 + 单页/多页分块 |
| DOC/DOCX | ✅ | python-docx |
| XLS/XLSX | ✅ | openpyxl + xlrd |
| PPT/PPTX | ✅ | python-pptx |
| TXT/MD | ✅ | 直接读取 |
| OFD | ✅ | ofdparser |
| 扫描件 PDF | ⚠️ | 需安装 PaddleOCR |
| DWG/DXF | ❌ | 图纸标注，非文本 |

## 配置文件

`config.yaml` 包含：
- **路径配置**: 知识库目录、数据库路径、模型目录
- **域配置**: 变电/配电/送电输电/综合的类目树
- **嵌入配置**: BGE-M3 本地 GPU 或 Ollama 回退
- **重排序配置**: FlagEmbedding 交叉编码器 或 嵌入相似度
- **LLM 配置**: Ollama / 百炼 / llama-cpp，`think: false` 关闭 Qwen3.5 思考链
- **检索参数**: RRF 融合、粗召回数、精排数、置信度校准

## API 端点

详见 [TECHNICAL.md](TECHNICAL.md) 或访问 `http://localhost:8000/docs`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/stats` | 知识库统计 |
| POST | `/upload` | 单文件上传入库 |
| POST | `/upload/batch` | 批量上传入库 |
| GET | `/files` | 列出已入库文件 |
| GET | `/files/{id}` | 文件详情 |
| GET | `/files/summary` | 入库统计摘要 |
| DELETE | `/files/{id}` | 删除文件 |
| POST | `/files/{id}/reindex` | 重建索引 |
| POST | `/search` | 纯检索 (含置信度) |
| POST | `/ask` | RAG 问答 (含多轮对话) |
| POST | `/ask/stream` | SSE 流式问答 |
| POST/GET/DELETE | `/conversations` | 会话管理 |
