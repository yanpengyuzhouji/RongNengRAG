# 部署指南

本文档说明如何完整部署 RAG 知识库系统，包括依赖服务准备、后端、前端。

## 总体架构

系统由 4 部分组成：

| 组件 | 技术 | 说明 |
|------|------|------|
| LLM 服务 | 任意 OpenAI 兼容服务 | vLLM / Xinference / LM Studio / PaddleX serving / 云厂商 |
| OCR-VL 服务 | PaddleOCR-VL (OpenAI 兼容) | 扫描件 PDF/图片的结构化识别 (可选) |
| 后端 | FastAPI + Milvus Lite + Qwen dense API + BGE-M3 sparse + BM25 + BGE-Reranker | 本机运行 |
| 前端 | Vue 3 + Element Plus | 本机运行, 默认 5178 |

LLM 和 OCR-VL 服务可以部署在局域网其他机器上，后端通过 `config.yaml` 指定其地址。

## 1. 准备 LLM 服务

后端通过 OpenAI 兼容协议 (`POST /v1/chat/completions`) 调用 LLM。任何提供该协议的服务均可使用。

**推荐: vLLM 本地部署**

```bash
pip install vllm
vllm serve Qwen/Qwen3-4B --port 18000 --host 0.0.0.0
```

验证:
```bash
curl http://<llm-ip>:18000/v1/models
```

**也可使用**: Xinference、LM Studio、Ollama（OpenAI 兼容端点 `/v1`）、PaddleX serving、阿里云百炼/OpenAI 云 API 等。

## 2. 准备 OCR-VL 服务 (可选)

如果知识库含扫描件 PDF、PNG/JPEG 等图片，需要 PaddleOCR-VL 服务做结构化识别。无扫描件可跳过，`ocr.enabled: false`。当前主链路使用 8001 的 Pipeline 接口；8080 兼容接口仅用于预览页中的 OCR 对比诊断。

**PaddleOCR-VL 部署**（PaddleX 方式，供参考）:

```bash
pip install paddlex
# 下载并启动 paddleocr-vl-1.6 模型 serving
paddlex --serve --model paddleocr_vl_1.6 --port 8001 --host 0.0.0.0
```

8001 Pipeline 验证:
```bash
curl http://<ocr-ip>:8001/openapi.json
```

当前 RAG 配置:
```json
{
  "ocr": {
    "enabled": true,
    "always_pipeline": true,
    "vl": {
      "base_url": "http://<ocr-ip>:8001",
      "protocol": "pipeline",
      "endpoint": "/ocr"
    },
    "compare": {
      "legacy_base_url": "http://<ocr-ip>:8080",
      "legacy_protocol": "openai"
    }
  }
}
```

8001 接收 multipart `POST /ocr`，返回包含 `parsing_res_list` 的版面块；8080 若启用对比功能，则提供 OpenAI 兼容的 `/v1/models` 与 `/v1/chat/completions`。具体模型部署方式以 PaddleOCR-VL 官方文档为准。

## 3. 部署后端

### 环境

- Python 3.10+
- NVIDIA GPU 可选。Qwen 稠密向量通过 API 获取；BGE-M3 稀疏向量和 BGE-Reranker 默认本地 CPU 运行，GPU 仅用于加速。

### 安装

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate  # Linux/macOS

pip install -r requirements.txt

# PyTorch: 按环境选
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA
# pip install torch                                                   # CPU
```

### 配置 `config.yaml`

```yaml
llm:
  provider: "openai"
  openai:
    base_url: "http://192.168.0.201:18000/v1"   # LLM 服务地址
    model: "/data/models/Qwen/Qwen3-4B"
    api_key: ""                                   # 云服务填 key

ocr:
  enabled: true
  always_pipeline: true
  provider: "vl"
  vl:
    base_url: "http://192.168.0.201:8001"          # Pipeline 服务地址
    protocol: "pipeline"
    endpoint: "/ocr"
  compare:
    enabled: true
    legacy_base_url: "http://192.168.0.201:8080"   # 可选，对比服务
    legacy_protocol: "openai"

paths:
  metadata_db: "data/file_metadata.db"             # SQLite 注册表
  milvus_db: "data/milvus_lite.db"                 # 向量库

embedding:
  provider: "openai"                                # Qwen dense API
  dimensions: 1024
  sparse_provider: "bge_m3"                         # 本地 BGE-M3 sparse
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

### 启动

```bash
cd src
uvicorn api.main:app --host 0.0.0.0 --port 8008
```

- API 文档: `http://<server-ip>:8008/docs`
- 健康检查: `GET /health` → `{"status":"ok",...}`

### 首次运行

- 自动下载本地 `BAAI/bge-m3`（仅生成 sparse lexical weights）和 `BAAI/bge-reranker-v2-m3`（交叉编码器重排），写入 `embedding.hf_home` 配置的缓存目录。
- Qwen 稠密向量不在本机下载，使用 `embedding.openai` 的 OpenAI-compatible `/v1/embeddings` 服务。
- 国内网络可配置 `hf_endpoint`；模型也可以预先下载到 `embedding.hf_home` 后离线启动。

### 当前检索链路

每个 chunk 入库时保留 Qwen 1024 维 dense vector，同时由本地 CPU BGE-M3 生成 Milvus `SPARSE_FLOAT_VECTOR`。查询阶段先执行 Milvus dense+BGE-M3 sparse 加权混合召回，再用当前知识库语料构建的 BM25（IDF + 文档长度归一化）补充关键词召回，最后对候选调用本地 BGE-Reranker-v2-m3 精排。配置位于 `config.yaml` 的 `embedding`、`reranker` 和 `retrieval` 段。

### 旧库稀疏向量迁移

如果知识库是在启用 BGE-M3 之前建立的，必须停掉后端后执行一次迁移。迁移只替换 `sparse_vector`，不重新请求 Qwen，也不改变文本、dense 向量和元数据：

```bash
python scripts/rebuild_sparse_vectors.py
```

迁移完成后再启动后端。后续新上传、编辑保存和重建索引会自动生成 BGE-M3 sparse vector。

## 4. 部署前端

```bash
cd src/ui-vue2
npm install

# 开发模式
npm run dev          # http://<server-ip>:5178

# 生产构建
npm run build        # 产物在 dist/
npm run preview
```

**后端不在本机时**，设置环境变量:

```bash
VITE_API_BASE=http://<server-ip>:8008 npm run dev
```

或创建 `src/ui-vue2/.env`:
```
VITE_API_BASE=http://<server-ip>:8008
```

## 5. 防火墙 / 网络

- 后端 8008 端口需对前端所在机器开放（含 SSE 流式长连接）。
- LLM/OCR-VL 服务端口（18000/8080 等）需对后端所在机器开放。
- 如通过浏览器访问前端，确保前端 → 后端跨域允许（后端已启用 CORS）。

## 6. 数据管理

- 入库文档、向量库、注册表全部在 `data/` 目录，无需额外数据库服务。
- 删除某文件:
  ```bash
  python scripts/build_index.py delete "<文件名或hash>"
  ```
- 查看统计:
  ```bash
  python scripts/build_index.py summary
  ```

## 7. 常见问题

**Q: LLM 返回 "LLM 服务不可用"**
检查 `config.yaml` 的 `llm.openai.base_url` 是否可达、模型名是否正确：
```bash
curl http://<llm-ip>:18000/v1/models
```

**Q: 扫描件入库后预览为空**
检查 8001 Pipeline 服务可达性、`ocr.vl.base_url`、`protocol` 和 `endpoint` 配置：
```bash
curl http://<ocr-ip>:8001/openapi.json
```

如果正文可检索但版面预览为空，检查 `data/parsed_cache/{file_hash}.layout.json` 是否存在。PDF 的版面缓存按页保存；修改渲染器后，重启后端并刷新预览即可重新生成 HTML，不需要重新生成向量。

**Q: PDF 预览页面过宽或出现双重滚动条**

当前预览由外层文档容器统一滚动，iframe 内部禁止滚动，并按页面原始宽度缩放到预览区。确认使用最新前端构建产物；若仍显示旧样式，清理浏览器缓存后重新加载。

**Q: 公式显示为原始 LaTeX**

公式渲染依赖预览 iframe 加载 MathJax CDN。检查浏览器是否能访问 `cdn.jsdelivr.net`；离线部署时需要改为本地 MathJax 资源。

**Q: 嵌入模型下载慢/失败**
配置 HF 镜像后重启：
```yaml
embedding:
  hf_endpoint: "https://hf-mirror.com"
  hf_home: "D:/git/RongNengRAG/data/hf_cache"
  sparse_provider: "bge_m3"
  sparse_device: "cpu"
```

**Q: CPU 重排太慢**

`bge-reranker-v2-m3` CPU 可以运行，但候选越多越慢。可降低 `retrieval.coarse_top_k` 或 `reranker.batch_size`，生产环境建议将 `reranker.device` 改为 `cuda`；切换设备不需要重新生成向量，只需重启后端。
