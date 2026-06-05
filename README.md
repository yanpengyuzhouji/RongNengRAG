# 榕能电力审图知识库 RAG 系统

基于 RAG (Retrieval-Augmented Generation) 架构的电力设计审图智能问答系统，服务于变电、配电、送电输电三大专业域的 50,000+ 电力设计文件知识库。

## 技术栈

| 层次 | 技术 |
|------|------|
| 向量数据库 | Milvus Lite (内嵌式，无需 Docker) |
| 嵌入模型 | BGE-M3 (`bge-m3:latest`，通过 Ollama 本地部署) |
| LLM | Qwen3 8B (`qwen3:8b`，通过 Ollama API 调用) |
| 重排序 | 嵌入余弦相似度 + 元数据加权 (Ollama) |
| 后端 API | FastAPI + Pydantic v2 |
| 前端 UI | Gradio 5.x (HTTP 客户端模式) |
| 文档解析 | PyMuPDF (PDF)、python-docx (DOCX)、python-pptx (PPTX)、openpyxl (XLSX) |
| OCR | PaddleOCR (可选，扫描件识别) |
| GPU 加速 | CUDA (RTX 4070 SUPER)，Ollama 自动调用 |

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

# 4. 拉取 Ollama 模型
ollama pull bge-m3:latest          # 嵌入模型
ollama pull qwen3:8b                # 问答模型 (或其他 Qwen 系列)

# 5. 创建数据目录
mkdir -p D:/rag-system/data/uploads D:/rag-system/data/parsed_cache D:/rag-system/models
cp config.yaml D:/rag-system/config.yaml
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
├── requirements.txt             # Python 依赖
├── README.md                    # 本文件
├── CHANGELOG.md                 # 开发日志
├── TECHNICAL.md                 # 技术文档
├── scripts/
│   └── build_index.py           # 命令行索引工具 (add-file/add-dir/list/delete/reindex)
└── src/
    ├── api/
    │   └── main.py              # FastAPI 后端 (REST API + 文件管理)
    ├── ui/
    │   └── app.py               # Gradio 前端 (HTTP 客户端，不直接操作 DB)
    ├── ingestion/               # 数据入库管道
    │   ├── file_processor.py    # 文件处理编排器 (parse→chunk→embed→insert)
    │   ├── file_walker.py       # 知识库目录遍历 + 元数据提取
    │   ├── chunker.py           # 格式感知分块引擎 (语义/单页/全文)
    │   ├── embedder.py          # 嵌入引擎 (Ollama BGE-M3)
    │   ├── milvus_store.py      # Milvus Lite 向量库封装
    │   └── pdf_parser.py        # PDF 文本提取 (PyMuPDF)
    ├── retrieval/               # 检索管道
    │   ├── retriever.py         # 三阶段检索编排器 (分析→召回→精排)
    │   ├── query_analyzer.py    # 查询分析器 (域分类+参数提取+同义词扩展)
    │   └── reranker.py          # 重排序器 (嵌入相似度+元数据加权)
    └── generation/              # 回答生成
        ├── llm_engine.py        # LLM 推理引擎 (Ollama/llama.cpp)
        └── prompt_templates.py  # 领域提示词模板
```

## 架构概览

```
                     ┌──────────────┐
                     │   Gradio UI  │  (浏览器交互)
                     │  :7860       │
                     └──────┬───────┘
                            │ HTTP API 调用
                     ┌──────▼───────┐
                     │   FastAPI    │  (后端服务)
                     │  :8000       │
                     └──┬───┬───┬──┘
          ┌─────────────┘   │   └─────────────┐
          ▼                 ▼                 ▼
   ┌──────────────┐  ┌──────────┐   ┌──────────────┐
   │ FileProcessor│  │ Retriever│   │  LLMEngine   │
   │ (入库管道)    │  │ (检索管道)│   │  (Ollama)    │
   └──────┬───────┘  └────┬─────┘   └──────────────┘
          │               │
          ▼               ▼
   ┌──────────────┐  ┌──────────┐
   │ Milvus Lite  │  │ Embedder │
   │ (向量库)     │  │ (BGE-M3) │
   └──────────────┘  └──────────┘
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
- **嵌入配置**: 选择 `ollama` 或 `sentence_transformers` 作为嵌入后端
- **重排序配置**: 选择 `ollama` (相似度) 或 `flagembedding` (交叉编码器)
- **LLM 配置**: 选择 `ollama` 或 `llama_cpp` 作为推理后端
- **检索参数**: RRF 融合参数、粗召回数、精排数

## API 端点

详见 [TECHNICAL.md](TECHNICAL.md) 或访问 `http://localhost:8000/docs`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/stats` | 知识库统计 |
| POST | `/upload` | 单文件上传入库 |
| POST | `/upload/batch` | 批量上传入库 |
| GET | `/files` | 列出已入库文件 |
| GET | `/files/summary` | 入库统计摘要 |
| DELETE | `/files/{id}` | 删除文件 |
| POST | `/files/{id}/reindex` | 重建文件索引 |
| POST | `/search` | 纯检索 (不生成回答) |
| POST | `/ask` | RAG 完整问答 |
