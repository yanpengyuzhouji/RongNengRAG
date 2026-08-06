# 部署指南

本文档说明如何完整部署 RAG 知识库系统，包括依赖服务准备、后端、前端。

## 总体架构

系统由 4 部分组成：

| 组件 | 技术 | 说明 |
|------|------|------|
| LLM 服务 | 任意 OpenAI 兼容服务 | vLLM / Xinference / LM Studio / PaddleX serving / 云厂商 |
| OCR-VL 服务 | PaddleOCR-VL (OpenAI 兼容) | 扫描件 PDF/图片的结构化识别 (可选) |
| 后端 | FastAPI + Milvus Lite + BGE-M3 | 本机运行 |
| 前端 | Vue 3 + Element Plus | 本机运行, 默认 5174 |

LLM 和 OCR-VL 服务可以部署在局域网其他机器上，后端通过 `config.yaml` 指定其地址。

## 1. 准备 LLM 服务

后端通过 OpenAI 兼容协议 (`POST /v1/chat/completions`) 调用 LLM。任何提供该协议的服务均可使用。

**推荐: vLLM 本地部署**

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-7B-Instruct --port 18000 --host 0.0.0.0
```

验证:
```bash
curl http://<llm-ip>:18000/v1/models
```

**也可使用**: Xinference、LM Studio、Ollama（OpenAI 兼容端点 `/v1`）、PaddleX serving、阿里云百炼/OpenAI 云 API 等。

## 2. 准备 OCR-VL 服务 (可选)

如果知识库含扫描件 PDF（无文字层），需要 PaddleOCR-VL 服务做结构化识别。无扫描件可跳过，`ocr.enabled: false`。

**PaddleOCR-VL 部署**（PaddleX 方式，供参考）:

```bash
pip install paddlex
# 下载并启动 paddleocr-vl-1.6 模型 serving
paddlex --serve --model paddleocr_vl_1.6 --port 8080 --host 0.0.0.0
```

验证:
```bash
curl http://<ocr-ip>:8080/v1/models
```

OCR-VL 服务需支持 OpenAI 兼容的图片输入:
```json
{
  "model": "paddleocr-vl-1.6",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "识别图中全部文字..."},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }]
}
```

> 具体部署方式以 PaddleOCR-VL 官方文档为准。只要协议是 OpenAI 兼容的 `/v1/chat/completions`，即可接入。

## 3. 部署后端

### 环境

- Python 3.10+
- 推荐 NVIDIA GPU（嵌入/重排加速）；无 GPU 用 CPU（较慢）

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
    model: "qwen2.5-7b-instruct"
    api_key: ""                                   # 云服务填 key

ocr:
  enabled: true
  provider: "vl"
  vl:
    base_url: "http://192.168.0.201:8080"          # OCR-VL 服务地址
    model: "paddleocr-vl-1.6"

paths:
  metadata_db: "data/file_metadata.db"             # SQLite 注册表
  milvus_db: "data/milvus_lite.db"                 # 向量库
```

### 启动

```bash
cd src
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- API 文档: `http://<server-ip>:8000/docs`
- 健康检查: `GET /health` → `{"status":"ok",...}`

### 首次运行

- 自动下载 BGE-M3 / BGE-Reranker 模型（写入 `hf_home` 配置的缓存目录）。
- 国内网络建议配置 `hf_endpoint: https://hf-mirror.com`（`config.yaml` embedding 段已有）。

## 4. 部署前端

```bash
cd src/ui-vue2
npm install

# 开发模式
npm run dev          # http://<server-ip>:5174

# 生产构建
npm run build        # 产物在 dist/
npm run preview
```

**后端不在本机时**，设置环境变量:

```bash
VITE_API_BASE=http://<server-ip>:8000 npm run dev
```

或创建 `src/ui-vue2/.env`:
```
VITE_API_BASE=http://<server-ip>:8000
```

## 5. 防火墙 / 网络

- 后端 8000 端口需对前端所在机器开放（含 SSE 流式长连接）。
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
检查 OCR-VL 服务可达性、`ocr.vl.base_url` 配置：
```bash
curl http://<ocr-ip>:8080/v1/models
```

**Q: 嵌入模型下载慢/失败**
配置 HF 镜像后重启：
```yaml
embedding:
  hf_endpoint: "https://hf-mirror.com"
```
