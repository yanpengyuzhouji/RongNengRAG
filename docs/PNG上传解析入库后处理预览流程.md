# PNG/PDF 上传、解析、入库与后处理预览流程

> 文档状态：当前实现说明
>
> 适用范围：PNG/JPEG 等图片与 PDF 文件从前端上传，到 OCR、分块、向量入库、版面缓存、目录导航和文档预览的完整链路。
>
> 代码基线：2026-08-11

## 1. 结论先行

当前 PNG 链路不是“图片直接入 Milvus”，而是同时生成两类产物：

1. **检索产物**：OCR 文本 → Chunk → Qwen 稠密向量 + 本地 CPU BGE-M3 稀疏向量 → Milvus；查询时融合 BM25 并使用本地 CPU 重排模型。
2. **视觉预览产物**：OCR 版面块（文本、bbox、block_type 等）→ `*.layout.json` → 预览时渲染成带坐标的 HTML iframe。

两条产物使用同一次 OCR 结果，但用途不同：

```text
前端选择 PNG
      │ multipart/form-data
      ▼
POST /upload
      │ 保存原图 + SHA256 + 文件注册
      ▼
FileProcessor.process
      │
      ├─ VL OCR 8001：返回纯文本和 parsing_res_list 版面块
      │       │
      │       ├─ 纯文本 → 清洗/去重 → 分块 → Embedding → Milvus
    │       └─ 版面块 → data/parsed_cache/{file_hash}.layout.json（图片为列表，PDF 按页保存）
      │
      └─ SQLite file_registry：状态、文件元数据、chunk 数、耗时

预览：
GET /files/{hash}/content
      │
      ├─ 从 Milvus 取 chunks/full_text
      └─ 从 layout.json 取版面块 → render_layout_html → layout_html
              │
              ▼
        DocumentPreview iframe（MathJax、表格、标题、公式、序号）
```

## 2. 主要代码位置

| 层次 | 文件 | 职责 |
|---|---|---|
| 前端上传 | `src/ui-vue2/src/KnowledgeBase.vue` | 选择文件、显示状态、逐个调用上传接口 |
| 前端 API | `src/ui-vue2/src/api.js` | API 地址、API Key、FormData、超时和错误处理 |
| 前端预览 | `src/ui-vue2/src/DocumentPreview.vue` | 请求内容、展示 iframe、OCR 对比、文本回退 |
| 上传接口 | `src/api/main.py` | 文件名/扩展名校验、保存到 uploads、调用处理器 |
| 文件处理 | `src/ingestion/file_processor.py` | SHA256、注册、解析、分块、嵌入、入库、缓存版面 |
| OCR 协议 | `src/ingestion/pdf_parser.py` | 复用 OCR 客户端，对单图执行 OCR |
| OCR 客户端 | `src/ingestion/vl_ocr.py` | 调用 8001 Pipeline、排序去重、版面 HTML 渲染 |
| 分块 | `src/ingestion/chunker.py` | 生成 `Chunk` 及页码/元数据 |
| 向量入库 | `src/ingestion/embedder.py`、`src/ingestion/milvus_store.py` | 生成向量并写入 Milvus |
| 文件元数据 | `FileProcessor` 内部 SQLite registry | 保存状态、路径、类型、标签和统计信息 |

## 3. PNG 前端上传流程

### 3.1 选择文件

`KnowledgeBase.vue` 的隐藏 `<input type="file">` 支持多选，允许扩展名包含：

```text
.pdf .doc .docx .xls .xlsx .ppt .pptx .wps .ofd .txt .md
.jpg .jpeg .png
```

选择后，前端只在内存中建立上传项：

```js
{
  file,
  status: 'pending',
  progress: 0,
  result: null,
  error: ''
}
```

当前上传窗口可以选择专业域、类目、子类目和文件类型，但普通文件上传接口实际只提交 `domain`、`category`；子类目和文件类型选择目前没有随 `/upload` 发送。

### 3.2 开始上传

`startUpload()` 按选择顺序逐个处理文件：

1. 将状态置为 `uploading`，进度先显示 20%。
2. 调用 `API.uploadFile(file, domain, category)`。
3. 成功后显示 100%，失败则保留错误信息并允许重试。
4. 全部完成后刷新文件列表和统计摘要。

当前 UI 使用的是单文件 `POST /upload`，不是批量 `POST /upload/batch`，所以多张 PNG 会顺序上传和顺序入库。

### 3.3 API 请求

`api.js` 使用 `FormData`：

```text
file     = 图片二进制
domain   = 可选
category = 可选
```

请求超时为 600 秒。API 地址默认跟随当前页面主机，生产环境可通过 `VITE_API_BASE` 覆盖；API Key 从 sessionStorage 或 `VITE_API_KEY` 读取，放在 `X-API-Key` 请求头中。

## 4. 后端接收与文件注册

### 4.1 `/upload` 接口

`src/api/main.py` 的 `POST /upload` 是同步接口，处理顺序为：

1. 校验文件名不为空。
2. `sanitize_upload_filename()` 清理文件名，防止路径穿越和非法文件名。
3. 校验扩展名；PNG、JPG、JPEG 等图片在白名单内。
4. 将原始文件写入 `config.yaml` 的 `paths.uploads_dir`，默认是 `data/uploads`。
5. 调用 `FileProcessor.process()`。
6. 返回成功状态、SHA256、chunk 数、字符数、元数据和耗时。

请求在处理完成前不会返回，因此 PNG OCR、嵌入和 Milvus 写入耗时都会占用这次 HTTP 请求。

### 4.2 SHA256 去重与注册表

`FileProcessor.process()` 首先计算文件 SHA256，并查询 SQLite `file_registry`：

- 已完成且已有版面缓存：直接返回“文件已入库，无需重复处理”。
- 已完成但图片没有 `*.layout.json`：强制重新处理，以补齐版面缓存。
- 新文件：注册为 `processing`。
- 处理完成且向量生成/激活成功：更新为 `completed`。
- 解析为空或发生异常：更新为 `failed`。

主要注册字段包括：

```text
file_hash, original_path, stored_path, file_name, file_size, file_type
status, chunks_count, chars_count, domain, category, doc_number
error_message, parse_time_ms, embed_time_ms, created_at, updated_at
```

## 5. PNG OCR 与文本解析

### 5.1 图片分支

`FileProcessor._parse_file()` 对以下扩展名走同一条图片 OCR 分支：

```text
.jpg .jpeg .png .tif .tiff .bmp .gif
```

处理逻辑：

1. 调用 `PDFParser.ocr_page_with_layout(file_path)`。
2. `PDFParser` 延迟初始化 `VLOcrClient`。
3. 当前配置使用 8001 PaddleOCR Pipeline：

```yaml
ocr:
  enabled: true
  vl:
    base_url: http://192.168.0.201:8001
    protocol: pipeline
    endpoint: /ocr
```

4. 读取 PNG 原始二进制并以 multipart `POST /ocr` 发送。
5. OCR 返回 `parsing_res_list` 后，同时保留：
   - `block_content`：用于文本入库。
   - `bbox`/`block_bbox`、`block_type` 等：用于版面预览。

### 5.2 OCR 文本后处理

`VLOcrClient._recognize_pipeline()` 对版面块进行：

- 优先按坐标排序，其次按 `block_order`/`block_id` 排序；
- 去除重复块；
- 将 OCR 返回的字面量 `\\n` 转换为换行，但避免误伤 `\\neq`、`\\nabla` 等 LaTeX 命令；
- 保留 Markdown 表格和 LaTeX 公式；
- 拼接为下游分块使用的纯文本。

图片 OCR 失败或 OCR 未启用时返回空文本，当前 PNG 分支没有本地图片文字提取兜底，因此最终会以“解析后无有效文本内容”失败。

另外，`max_image_dim` 当前主要用于 PDF 页面渲染路径；独立 PNG 直接发送原始字节，未在本地按该配置缩放。

## 6. 分块、嵌入和向量入库

### 6.1 图片分块

PNG OCR 文本通过 `_chunk_text_safe()` 进入 `Chunker.chunk_text_document()`：

- 短文本：生成一个 `full_document` chunk；
- 长文本：按语义分隔符递归分块，并保留 overlap；
- 图片作为单文档处理，当前 chunk 的 `page_num` 通常为空，`total_pages` 为 1。

chunk 携带文件标签：专业域、类目、文档编号、发布层级、电压等级、专业类型、设备类型、年份、区域、文件路径和文件类型等。

### 6.2 嵌入和 Milvus

`_embed_and_insert()` 以每批 20 个 chunk 的方式处理：

1. `create_text_for_embedding()` 将元数据和正文组合成嵌入文本。
2. `Embedder.encode()` 保持 Qwen3-Embedding-0.6B API 生成 dense vector，并由本地 CPU `BAAI/bge-m3` 生成 lexical sparse weights；两者分别写入 Milvus 的 dense/sparse 字段。
3. `MilvusStore.begin_file_generation(file_hash)` 创建影子 generation。
4. 分批写入 chunk、文本、向量和元数据。
5. 校验 chunk 总数。
6. 激活 generation 对应的活动别名。
7. 成功后才把 SQLite registry 标记为 `completed`。

这保证重建索引时，旧活动集合不会在新索引验证前被破坏。

### 6.3 查询时的 BM25 与重排

Milvus dense+BGE-M3 sparse 负责第一阶段语义和词法召回；系统同时从当前 active generation 的 `text`、`embedding_text`、`file_path` 和 `doc_number` 构建 BM25 索引。BM25 包含语料 IDF 和文档长度归一化，用于补充标准号、文件名和关键词的精确命中。融合后的候选再交给本地 `BAAI/bge-reranker-v2-m3` CPU 交叉编码器重排。该重排模型只参与查询，不影响入库向量。

## 7. 版面缓存与预览后处理

### 7.1 版面缓存

PNG OCR 成功后，如果存在版面块，写入：

```text
data/parsed_cache/{file_hash}.layout.json
```

PNG 的缓存通常是一个 block 列表；PDF 使用以 0-based 页索引为键的字典。缓存保存原始 OCR 版面信息，不直接保存最终 HTML，因此渲染器更新后已有文档也能重新生成预览。

### 7.2 `/files/{identifier}/content`

前端打开预览时，`DocumentPreview.vue` 请求：

```text
GET /files/{file_hash}/content
```

后端执行：

1. 从 SQLite registry 按 hash 或文件名定位文件。
2. 从 Milvus 按 `file_hash` 取 chunks，并按页码/chunk 顺序排序。
3. 通过 `Retriever.get_full_document()` 拼接 `full_text`。
4. 对历史 chunk 再次执行异常 HTML 清洗和重复段落去重。
5. 读取 `*.layout.json`。
6. 调用 `render_layout_html(layout_blocks)` 生成 `layout_html`。
7. 返回文件元数据、`full_text`、chunks、layout blocks、layout HTML、chunk/page 统计。

### 7.3 `render_layout_html()`

版面渲染器位于 `src/ingestion/vl_ocr.py`，缓存预览和实时 OCR 对比共用同一实现，避免前后两套坐标逻辑产生差异。

当前后处理包括：

- 使用 bbox 生成绝对定位的 `.ocr-block`；
- 普通文本扩展到页面右边界，避免 OCR 框过窄导致提前换行；
- 表格保留被动 HTML 结构，并清理脚本、事件和外部资源属性；
- `title`/`heading`/`paragraph_title` 等标题使用整页宽度并居中；
- 公式交给 MathJax 渲染；
- 公式编号与公式建立关联，typeset 后重新垂直对齐；
- 公式和序号之间自动生成并重新定位 `---` 连接线；
- `content`/`contents`/`toc` 目次块按“标题、点引导线、页码”三列渲染，支持 `1`、`1.1` 等层级缩进；OCR 未标记目次时，按多行点线和页码特征自动识别；
- 目录候选综合 OCR 标题标签、章节编号、文字长度、标点、bbox 居中程度和页尾位置，过滤目次页码、页眉页脚及普通编号正文；
- `settle()` 在字体和 MathJax 完成后处理块的纵向碰撞。

iframe 使用 `sandbox="allow-scripts"`，因为 MathJax 需要脚本执行；OCR 内容在进入 iframe 前会经过清洗。

### 7.4 目录导航与前端预览优先级

后端同时返回 `outline`。目录不再按 PDF 页简单罗列，而是优先展示章节标题及层级：

1. 从布局块提取带 `id/anchor/page/level/title` 的标题候选；
2. 前端按 `level` 缩进显示，并保留页码提示；
3. 点击章节时定位对应 PDF 页，再通过 `postMessage` 让 iframe 滚动到标题锚点；
4. 没有可靠章节标题时，回退为页级目录。

预览页面使用固定的单页 iframe：页面按原始 bbox 计算宽度，在可用区域内缩放；iframe 内部隐藏滚动条，由外层文档区域统一滚动，页间只保留很小的分隔。这样 PDF 多页与 PNG 单页共享同一套视觉规则。

`DocumentPreview.vue` 的主区域按以下顺序选择内容：

1. `/files/{hash}/content` 返回的 `layout_html`：主预览。
2. 老后端只返回 `layout_blocks` 时，前端兼容渲染 `renderCachedLayout()`。
3. 没有版面缓存时，按 chunks 聚合页面，使用清洗后的文本/表格 HTML 回退预览。

主预览不默认显示两套 OCR；点击“OCR 对比”才调用 8001 Pipeline 和 8080 兼容 OCR，并按页展示诊断结果。

工具栏的“OCR 对比”是诊断功能，调用：

```text
GET /files/{hash}/ocr-compare
```

它会重新读取源文件：图片直接识别；PDF 按页渲染为 PNG，然后分别调用 8001 Pipeline 和 8080 兼容接口，结果只返回给前端，不写入向量库或版面缓存。

## 8. 当前 PNG 链路的边界

1. **上传是同步的**：大图 OCR 和嵌入期间请求一直占用连接，前端进度目前是阶段性展示，不是服务端实时进度。
2. **UI 批量实际串行**：文件多时总耗时线性增加。
3. **图片没有真实页码语义**：文本回退路径的 `page_num` 通常为 0/空，主要依赖版面 iframe 展示。
4. **版面缓存没有渲染版本号**：接口会用当前渲染器重新生成 HTML，但旧 JSON 的字段兼容性仍需保证。
5. **独立 PNG 无 OCR 兜底**：8001 不可用时不会自动转本地 OCR 引擎。
6. **预览编辑不落库**：当前“编辑”只修改前端内存和编辑记录，刷新页面即丢失。
7. **OCR 对比会重新消耗 OCR 服务**：它不是缓存读取接口，不能作为普通预览主链路。
8. **MathJax 依赖外部 CDN**：无外网环境下公式可能只显示原始 LaTeX 或不完成 typeset。
9. **OCR 质量仍受外部模型影响**：目次样式已统一，但异常编号、缺失章节号或严重错字仍可能导致目录漏项；无法可靠识别时会回退页级导航。

## 9. 当前 PDF 链路（已实现）

PDF 和 PNG 共用前端上传入口、`POST /upload`、文件注册、Embedding、Milvus 和预览接口，但 PDF 在“解析”和“页面组织”上多了一层。

### 9.1 当前实际流程

```text
前端选择 PDF
      │
      ▼
POST /upload
      │ 保存 PDF + SHA256 + file_registry
      ▼
FileProcessor.process
      │
      ├─ OCR 开启且不是图纸：_process_pdf_progressive
      │       │
      │       ├─ fitz 逐页提取原生文本并判断 needs_ocr
      │       ├─ 需要 OCR 的页（always_pipeline=true 时为全部页）
      │       │     └─ fitz 页面渲染 PNG → 8001 Pipeline
      │       ├─ 每页选择 OCR 文本或 PDF 原文回退
      │       ├─ 按页分块，保留 page_num
      │       └─ 保存按页 layout cache
      │
      └─ OCR 关闭：_parse_file → PDFParser.parse → chunk_pdf_document
              （只有文本分块，没有 OCR 版面缓存）
      │
      ▼
分批 Embedding → Milvus 影子集合 → 校验 → 激活
      │
      ▼
GET /files/{hash}/content
      ├─ Milvus chunks 按 page_num/chunk_index 排序
      ├─ full_text 加入页分隔符
      └─ layout cache → 当前 render_layout_html
              │
              ▼
DocumentPreview：当前 layout_html 优先，否则文本分页回退
```

### 9.2 PDF 解析阶段

`PDFParser.parse()` 当前使用 PyMuPDF（fitz）做两阶段判断：

1. 最多采样前 15 页。
2. 单页提取文本少于 `ocr.min_text_chars`（默认 50）时标记 `needs_ocr`。
3. 采样页中至少 70% 需要 OCR 时，判断为全扫描 PDF，并跳过剩余页面的无效 fitz 文本提取。
4. 否则继续逐页提取；如果最终超过 50% 页面需要 OCR，则将文档标记为扫描文档。

随后，`ocr.always_pipeline=true` 会把 OCR 请求目标覆盖为全部页面。当前默认配置因此会对每页调用 8001 Pipeline，但文本入库仍按页面原先的 `needs_ocr` 状态选择：

- 原生文本充足的页面：入库优先使用 fitz 文本；
- 原生文本不足的页面：使用有效 OCR 文本；
- OCR 为空或被垃圾检测拒绝：回退到该页原生 PDF 文本。

### 9.3 PDF OCR 阶段

PDF 不能像 PNG 那样直接把源文件作为图片发送。每个 OCR 页面会经过：

```text
PDF page
  → fitz.get_pixmap(dpi=150)
  → 超过 max_image_dim 时缩放到最大边 3000px
  → PNG bytes
  → 8001 POST /ocr
```

`VLOcrClient.recognize_pdf_pages()` 返回两套信息：

- `page_texts[page_index]`：每页结构化 OCR 文本；
- `last_layout_pages[page_index]`：每页 `parsing_res_list` 版面块。

其中 `page_index` 在内部和缓存中以 0-based 为主；生成 chunk 时转换为用户可见的 1-based `page_num`。

### 9.4 PDF 分块和入库

PDF 使用 `Chunker.chunk_page_text()`，与 PNG 的 `chunk_text_document()` 不同：

- 短页：一页一个 chunk；
- 长页：只在页内语义拆分，不跨页切分；
- chunk ID 形如 `{file_hash}_p{page_num}_{index}`；
- 每个 chunk 写入 `page_num`、`total_pages` 和 `chunk_index`。

后续 `_embed_and_insert()` 与 PNG 完全相同：每批 20 个 chunk，生成稠密/稀疏向量，写入影子 collection，校验后切换活动别名，再提交 registry 的 `completed` 状态。

### 9.5 PDF 版面缓存和当前预览行为

PDF OCR 成功后，当前缓存大致为：

```json
{
  "0": ["第 1 页的 blocks"],
  "1": ["第 2 页的 blocks"]
}
```

后端现在保留按页分组，并返回 `layout_pages`：每个页面单独调用与 PNG 相同的 `render_layout_html()`，生成一个独立的 `layout_html`。`layout_html` 字段仍保留，用于单页 PNG、单页 PDF 和旧前端兼容；同时返回基于章节的 `outline`。

前端现在按 `layout_pages` 纵向展示多个独立 iframe；每个 iframe 内部都是一个完整的 PNG 风格页面，因此不同 PDF 页的 bbox 不再共享同一坐标页面：

- **文本回退预览**：可以根据 Milvus 的 `page_num` 生成多页块和目录；
- **layout_html 主预览**：多页 PDF 使用多个独立 iframe，每页保持自己的坐标、宽度和高度；
- **目录跳转**：优先按章节标题和层级导航，点击后定位页内 bbox 锚点；无可靠标题时才回退到 `page-{n}` 页级导航；
- **兼容回退**：旧服务只返回扁平 `layout_blocks` 时，前端会按 block 上的 `page` 字段重新分组。

`/files/{hash}/ocr-compare` 现在对 PDF 的每一页分别调用 8001 Pipeline 和 8080 兼容 OCR，返回 `pages[]`，前端按页显示两套识别结果；PNG 仍返回一个页面。OCR 对比只在用户点击按钮后执行，不影响已入库向量。

### 9.6 PNG 与 PDF 当前链路对照

| 环节 | PNG | PDF |
|---|---|---|
| 上传入口 | `POST /upload` | 相同 |
| 原始文件 | 直接保存图片 | 直接保存 PDF |
| OCR 输入 | 原始图片 bytes | 每页 PDF 先用 fitz 渲染成 PNG |
| OCR 次数 | 1 次 | 每个 OCR 页 1 次；默认 Pipeline 可能覆盖全部页 |
| 原生文本 | 没有 | 有，fitz 可提取 |
| OCR 回退 | OCR 失败通常无文本 | OCR 无效时可回退到该页 PDF 原文 |
| 版面缓存 | block 列表 | 按页 block 字典 |
| 分块方法 | 文档级 `chunk_text_document` | 页级 `chunk_page_text` |
| 页码 | 通常为空/0 | 1-based `page_num` |
| chunk ID | `{hash}_0` 或语义分块 ID | `{hash}_p{page}_{index}` |
| 向量入库 | 相同的分批/影子集合流程 | 相同 |
| 文本预览 | 可按单页回退 | 可按页回退 |
| 版面主预览 | 单页模型与 PNG 匹配 | 每页独立 iframe，使用同一单页渲染器 |
| OCR 对比 | 识别整张图片 | 逐页诊断整份 PDF |

## 10. PDF 后续优化规划

### 10.1 已完成基础

当前 PDF 上传、解析、入库、版面缓存、多页预览和 OCR 对比主链路已经闭环：

- 前端允许选择 PDF；
- `/upload` 白名单允许 `.pdf`；
- `PDFParser.parse()` 可用 PyMuPDF 逐页提取文本；
- 可按采样页判断扫描 PDF，并标记 `needs_ocr_pages`；
- `ocr.always_pipeline=true` 时可让所有页面走 8001 Pipeline；
- `_process_pdf_progressive()` 已按页 OCR、按页分块并保留 `page_num`；
- PDF OCR 版面块可写入同一个 `*.layout.json` 缓存。

因此，后续重点是性能、缓存 schema 和异常文档覆盖率优化，不需要重新设计上传入口。

### 10.2 当前目标链路

```text
PDF 上传
  → 文件校验、保存、SHA256、注册
  → fitz 读取页数/尺寸/文本
  → 每页判断：原生文本或 OCR
  → 每页输出 PageArtifact
       ├─ page_text
       ├─ page_num
       ├─ page_width/page_height
       ├─ text_source: fitz | ocr | fallback
       └─ layout_blocks
  → 页面级 chunk，chunk_id 带页码
  → 分批嵌入、影子集合写入、校验后激活
  → 保存 page-aware layout cache
  → 多页版面 HTML/iframe 预览
```

### 10.3 后续可选后端优化

1. **统一页面产物模型**：不要只保存裸 block；同时保存页码、页面宽高、OCR 状态、文本来源和渲染版本。
2. **版面缓存按页保存**：推荐结构：

   ```json
   {
     "schema_version": 1,
     "file_hash": "...",
     "pages": {
       "1": {
         "width": 1700,
         "height": 2400,
         "text_source": "ocr",
         "blocks": []
       }
     }
   }
   ```

3. **PDF 页面渲染参数统一**：fitz 渲染 DPI、最大尺寸、旋转角度、页面尺寸必须进入缓存，确保 OCR 坐标和预览坐标一致。
4. **OCR 回退明确化**：每页按 `fitz → OCR → 原生文本回退` 选择，并记录原因，避免整份 PDF 因单页 OCR 失败而丢文本。
5. **版面渲染器支持多页**：已完成。PDF 按页调用 PNG 同款渲染器，前端用多个 iframe 展示。
6. **接口返回章节目录**：已完成。`/files/{hash}/content` 返回 `layout_pages` 和 `outline`，前端按章节锚点跳转，无法识别时回退页级导航。
7. **缓存与索引解耦**：PDF 重新渲染预览不应重复生成向量；只缺 layout cache 时允许只补版面缓存。

### 10.4 后续可选前端优化

1. 上传组件沿用现有 PDF 选择逻辑，但补充文件大小、页数/预计耗时和服务端阶段进度展示。
2. 预览主区域改成真正的多页文档：已完成基础版，每页独立 iframe，目录按章节跳转。
3. 多页版面预览与文本回退都使用统一页模型。
4. OCR 对比已改为覆盖整份 PDF；后续可再增加单页选择，减少诊断耗时。
5. 大 PDF 使用按页懒加载，避免一次性把全部 iframe/HTML 注入 DOM。
6. 页面缩放应以页面原始宽高为基准，避免 bbox 坐标因 CSS 缩放出现错位。

### 10.5 后续推荐顺序

1. 为 layout cache 增加 schema/render version 和页面宽高元数据。
2. 对大 PDF 增加 iframe 懒加载和按页 OCR 诊断。
3. 增加异常章节编号、混合文本/扫描页和低质量 OCR 的目录回归样本。
4. 最后再补充大文件异步任务、断点/重试和服务端进度。

## 11. 验收清单

### PNG 当前链路

- [x] PNG 可以从前端选择并上传。
- [x] 后端保存原图并生成稳定 SHA256。
- [x] 8001 OCR 返回文本时可以完成分块、嵌入和 Milvus 入库。
- [x] `data/parsed_cache/{hash}.layout.json` 存在且可被预览接口读取。
- [x] 标题、表格、公式、公式编号、连接线和目次行在预览中保持版面关系。
- [x] 8001 失败时错误状态和回退行为明确。
- [x] 重复上传同一 PNG 不重复写入；缺失版面缓存时可补建。

### PDF 当前链路

- [x] 原生文本 PDF、扫描 PDF、混合 PDF 都能按页处理。
- [x] 每个 chunk 的 `page_num` 与版面缓存页码一致。
- [x] OCR 失败页保留可检索的原生 PDF 文本（若存在）。
- [x] 多页预览不会把不同页面的 bbox 混在同一页面。
- [x] 页面缩放、目录章节跳转、OCR 对比和文本回退一致。
- [x] 版面 HTML 在读取布局缓存时独立重建，不强制重做向量。
